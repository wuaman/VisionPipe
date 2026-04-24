#include "core/node_base.h"

#include <chrono>

#include "core/logger.h"

namespace visionpipe {

NodeBase::NodeBase(const std::string& name)
    : name_(name)
    , input_queue_(nullptr)
    , output_queue_(nullptr)
    , state_(NodeState::INIT) {}

NodeBase::~NodeBase() {
    stop(false);
    wait_stop();
}

NodeBase::NodeBase(NodeBase&& other) noexcept
    : name_(std::move(other.name_))
    , input_queue_(other.input_queue_)
    , output_queue_(std::move(other.output_queue_))
    , state_(other.state_.load())
    , processed_count_(other.processed_count_.load())
    , error_count_(other.error_count_.load())
    , params_(std::move(other.params_))
    , last_frame_time_(other.last_frame_time_.load())
    , frames_since_last_fps_(other.frames_since_last_fps_.load())
    , current_fps_(other.current_fps_.load()) {
    other.input_queue_ = nullptr;
}

NodeBase& NodeBase::operator=(NodeBase&& other) noexcept {
    if (this != &other) {
        stop(false);
        wait_stop();

        name_ = std::move(other.name_);
        input_queue_ = other.input_queue_;
        output_queue_ = std::move(other.output_queue_);
        state_ = other.state_.load();
        processed_count_ = other.processed_count_.load();
        error_count_ = other.error_count_.load();
        params_ = std::move(other.params_);
        last_frame_time_ = other.last_frame_time_.load();
        frames_since_last_fps_ = other.frames_since_last_fps_.load();
        current_fps_ = other.current_fps_.load();
        other.input_queue_ = nullptr;
    }
    return *this;
}

void NodeBase::create_output_queue(size_t capacity, OverflowPolicy policy) {
    output_queue_ = std::make_shared<BoundedQueue<Frame>>(capacity, policy);
}

bool NodeBase::set_param(const std::string& name, const ParamValue& value) {
    std::lock_guard<std::mutex> lock(params_mutex_);
    params_[name] = value;
    return true;
}

void NodeBase::start() {
    if (state_ == NodeState::RUNNING) {
        return;
    }

    state_ = NodeState::RUNNING;
    on_init();

    // 如果不是 SourceNode，启动工作线程从 input_queue 消费
    if (!is_source()) {
        worker_thread_ = std::thread(&NodeBase::worker_loop, this);
    }

    VP_LOG_INFO("Node '{}' started", name_);
}

void NodeBase::stop(bool drain) {
    NodeState expected = NodeState::RUNNING;
    if (!state_.compare_exchange_strong(expected, NodeState::DRAINING)) {
        // 不在 RUNNING 状态，直接停止
        state_ = NodeState::STOPPED;
        if (input_queue_) {
            input_queue_->stop();
        }
        return;
    }

    if (drain) {
        VP_LOG_INFO("Node '{}' draining...", name_);
    } else {
        state_ = NodeState::STOPPED;
        if (input_queue_) {
            input_queue_->stop();
        }
    }

    on_stop();
}

void NodeBase::wait_stop() {
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void NodeBase::worker_loop() {
    VP_LOG_DEBUG("Node '{}' worker thread started", name_);

    while (state_ != NodeState::STOPPED) {
        // DRAINING 状态下：队列为空则退出
        if (state_ == NodeState::DRAINING) {
            if (!input_queue_ || input_queue_->empty()) {
                state_ = NodeState::STOPPED;
                break;
            }
        }

        if (!input_queue_) {
            break;
        }

        // 尝试从 input_queue 取帧，带超时避免死锁
        auto frame_opt = input_queue_->pop_for(std::chrono::milliseconds(100));
        if (!frame_opt.has_value()) {
            // DRAINING + 队列空 → 退出
            if (state_ == NodeState::DRAINING) {
                state_ = NodeState::STOPPED;
                break;
            }
            continue;
        }

        Frame frame = std::move(*frame_opt);
        if (process_frame(frame)) {
            // 处理成功，推送到 output_queue
            if (output_queue_) {
                output_queue_->push(std::move(frame));
            }
        }
    }

    if (input_queue_) {
        input_queue_->stop();
    }
    VP_LOG_INFO("Node '{}' worker thread stopped, processed {} frames",
                name_, processed_count_.load());
}

bool NodeBase::process_frame(Frame& frame) {
    try {
        process(frame);
        ++processed_count_;

        // 更新帧率：记录窗口起始时间，每 10 帧计算一次
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        ++frames_since_last_fps_;

        int64_t zero = 0;
        last_frame_time_.compare_exchange_strong(zero, now);  // 窗口首帧初始化

        if (frames_since_last_fps_ >= 10) {
            int64_t window_start = last_frame_time_.exchange(now);
            if (window_start > 0) {
                double elapsed_sec = static_cast<double>(now - window_start) / 1e9;
                if (elapsed_sec > 0) {
                    current_fps_ = frames_since_last_fps_.load() / elapsed_sec;
                }
            }
            frames_since_last_fps_ = 0;
        }

        return true;
    } catch (const std::exception& e) {
        ++error_count_;
        on_error(frame, e.what());
        return false;
    } catch (...) {
        ++error_count_;
        on_error(frame, "unknown error");
        return false;
    }
}

void NodeBase::on_error(Frame& frame, const std::string& error) {
    VP_LOG_ERROR("Node '{}' error processing frame {}: {}", name_, frame.frame_id, error);
}

NodeStats NodeBase::stats() const {
    NodeStats s;
    s.processed_count = processed_count_.load();
    s.error_count = error_count_.load();
    s.fps = current_fps_.load();
    if (input_queue_) {
        s.input_queue_stats = input_queue_->stats();
    }
    return s;
}

}  // namespace visionpipe