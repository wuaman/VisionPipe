#include "core/infer_node.h"

#include <chrono>
#include <utility>

#include "core/error.h"
#include "core/logger.h"

namespace visionpipe {

namespace {
constexpr size_t kQueueCapacity = 1024;
}

InferNode::InferNode(std::shared_ptr<IModelEngine> engine,
                     size_t workers,
                     const std::string& name)
    : NodeBase(name)
    , engine_(std::move(engine))
    , workers_(workers == 0 ? 1 : workers)
    , owned_input_queue_(std::make_shared<BoundedQueue<Frame>>(kQueueCapacity, OverflowPolicy::BLOCK)) {
    if (!engine_) {
        throw ConfigError("InferNode requires a valid engine");
    }
    input_queue_ = owned_input_queue_.get();
    create_output_queue(kQueueCapacity, OverflowPolicy::BLOCK);
}

InferNode::~InferNode() {
    stop(false);
    wait_stop();
}

void InferNode::process(Frame& frame) {
    if (contexts_.empty()) {
        throw InferError("InferNode is not started");
    }

    Tensor output;
    contexts_.front()->infer(frame.image, output);
}

void InferNode::start() {
    if (state_ == NodeState::RUNNING) {
        return;
    }

    if (!input_queue_) {
        throw ConfigError("InferNode requires an input queue");
    }

    contexts_.clear();
    contexts_.reserve(workers_);
    for (size_t i = 0; i < workers_; ++i) {
        auto context = engine_->create_context();
        if (!context) {
            throw InferError("IModelEngine::create_context returned null");
        }
        contexts_.push_back(std::move(context));
    }

    processed_count_ = 0;
    error_count_ = 0;
    last_frame_time_ = 0;
    frames_since_last_fps_ = 0;
    current_fps_ = 0.0;

    {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        pending_outputs_.clear();
        next_output_frame_id_ = 0;
        next_output_initialized_ = false;
    }
    in_flight_frames_ = 0;

    state_ = NodeState::RUNNING;
    on_init();

    worker_threads_.clear();
    worker_threads_.reserve(workers_);
    for (size_t i = 0; i < workers_; ++i) {
        worker_threads_.emplace_back(&InferNode::worker_loop, this, i);
    }

    VP_LOG_INFO("InferNode '{}' started with {} worker(s)", name_, workers_);
}

void InferNode::stop(bool drain) {
    NodeState expected = NodeState::RUNNING;
    if (!state_.compare_exchange_strong(expected, NodeState::DRAINING)) {
        if (state_ == NodeState::INIT || state_ == NodeState::STOPPED) {
            return;
        }
    }

    if (!drain) {
        state_ = NodeState::STOPPED;
        if (input_queue_) {
            input_queue_->stop();
        }
        if (output_queue_) {
            output_queue_->stop();
        }
    }

    on_stop();
}

void InferNode::wait_stop() {
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();
    contexts_.clear();
}

void InferNode::worker_loop(size_t worker_index) {
    auto& context = contexts_.at(worker_index);

    while (!should_worker_exit()) {
        auto frame_opt = input_queue_->pop_for(std::chrono::milliseconds(100));
        if (!frame_opt.has_value()) {
            if (state_ == NodeState::DRAINING && input_queue_->empty()) {
                break;
            }
            continue;
        }

        Frame frame = std::move(*frame_opt);
        in_flight_frames_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(reorder_mutex_);
            if (!next_output_initialized_) {
                next_output_frame_id_ = frame.frame_id;
                next_output_initialized_ = true;
            }
        }

        try {
            Tensor output;
            context->infer(frame.image, output);
            ++processed_count_;

            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            std::lock_guard<std::mutex> fps_lock(fps_mutex_);
            ++frames_since_last_fps_;
            if (last_frame_time_ == 0) {
                last_frame_time_ = now;
            }
            if (frames_since_last_fps_ >= 10) {
                const int64_t window_start = last_frame_time_.exchange(now);
                if (window_start > 0) {
                    const double elapsed_sec = static_cast<double>(now - window_start) / 1e9;
                    if (elapsed_sec > 0.0) {
                        current_fps_ = frames_since_last_fps_.load() / elapsed_sec;
                    }
                }
                frames_since_last_fps_ = 0;
            }
        } catch (const std::exception& error) {
            ++error_count_;
            in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
            on_error(frame, error.what());
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(reorder_mutex_);
            pending_outputs_.emplace(frame.frame_id, std::move(frame));
            emit_ready_frames_locked();
        }
        in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
    }

    {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        emit_ready_frames_locked();
        if (state_ == NodeState::DRAINING && input_queue_ && input_queue_->empty() &&
            pending_outputs_.empty() && in_flight_frames_.load(std::memory_order_relaxed) == 0) {
            state_ = NodeState::STOPPED;
            if (output_queue_) {
                output_queue_->stop();
            }
        }
    }
}

void InferNode::emit_ready_frames_locked() {
    while (next_output_initialized_) {
        auto it = pending_outputs_.find(next_output_frame_id_);
        if (it == pending_outputs_.end()) {
            break;
        }

        if (output_queue_) {
            output_queue_->push(std::move(it->second));
        }
        pending_outputs_.erase(it);
        ++next_output_frame_id_;
    }
}

bool InferNode::should_worker_exit() const {
    if (state_ == NodeState::STOPPED) {
        return true;
    }
    if (state_ == NodeState::DRAINING && input_queue_ && input_queue_->empty()) {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        return pending_outputs_.empty() && in_flight_frames_.load(std::memory_order_relaxed) == 0;
    }
    return false;
}

}  // namespace visionpipe
