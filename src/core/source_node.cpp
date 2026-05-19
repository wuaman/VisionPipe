#include "core/source_node.h"

#include <chrono>

#include "core/error.h"
#include "core/logger.h"

namespace visionpipe {

SourceNode::SourceNode(const std::string& name, const SourceConfig& config)
    : NodeBase(name), config_(config) {
    create_output_queue(config_.queue_capacity, config_.overflow_policy);
}

SourceNode::~SourceNode() {
    if (!pipeline_managed_stop_) {
        stop(false);
    }
    if (source_thread_.joinable()) {
        source_thread_.join();
    }
}

SourceNode::SourceNode(SourceNode&& other) noexcept
    : NodeBase(std::move(other))
    , config_(std::move(other.config_))
    , pipeline_managed_stop_(other.pipeline_managed_stop_.load()) {}

SourceNode& SourceNode::operator=(SourceNode&& other) noexcept {
    if (this != &other) {
        if (!pipeline_managed_stop_) {
            stop(false);
        }
        if (source_thread_.joinable()) {
            source_thread_.join();
        }

        NodeBase::operator=(std::move(other));
        config_ = std::move(other.config_);
        pipeline_managed_stop_ = other.pipeline_managed_stop_.load();
    }
    return *this;
}

void SourceNode::process(Frame& frame) {
    (void)frame;
}

void SourceNode::start() {
    if (state_ == NodeState::RUNNING) {
        return;
    }

    on_open();

    state_ = NodeState::RUNNING;
    source_thread_ = std::thread(&SourceNode::source_worker_loop, this);
}

void SourceNode::stop(bool drain) {
    NodeState expected = NodeState::RUNNING;
    if (!state_.compare_exchange_strong(expected, NodeState::DRAINING)) {
        state_ = NodeState::STOPPED;
        return;
    }

    if (!drain) {
        state_ = NodeState::STOPPED;
    }

    if (output_queue_) {
        output_queue_->stop();
    }
}

void SourceNode::wait_stop() {
    if (source_thread_.joinable()) {
        source_thread_.join();
    }
}

void SourceNode::on_read_error(const std::exception& e) {
    VP_LOG_ERROR("SourceNode '{}' read error: {}", name_, e.what());
}

void SourceNode::source_worker_loop() {
    VP_LOG_DEBUG("SourceNode '{}' worker thread started", name_);

    Frame frame;
    frame.stream_id = config_.stream_id;
    int64_t frame_counter = 0;
    int64_t skip_counter = 0;

    while (state_ == NodeState::RUNNING) {
        bool read_ok = false;
        bool had_exception = false;
        try {
            read_ok = read_next(frame);
        } catch (const std::exception& e) {
            on_read_error(e);
            ++error_count_;
            had_exception = true;
        }

        if (!read_ok) {
            if (had_exception && config_.max_retries > 0) {
                bool recovered = false;
                for (int attempt = 1; attempt <= config_.max_retries; ++attempt) {
                    if (state_ != NodeState::RUNNING) break;
                    VP_LOG_INFO("SourceNode '{}' retry {}/{}", name_, attempt, config_.max_retries);
                    std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_interval_ms));
                    if (state_ != NodeState::RUNNING) break;

                    on_close();
                    try {
                        on_open();
                    } catch (const std::exception& e) {
                        VP_LOG_WARN("SourceNode '{}' reopen failed on retry {}: {}", name_, attempt, e.what());
                        continue;
                    }

                    frame = Frame();
                    frame.stream_id = config_.stream_id;
                    skip_counter = 0;
                    recovered = true;
                    break;
                }
                if (recovered) continue;
                break;
            }

            if (config_.loop) {
                VP_LOG_INFO("SourceNode '{}' looping back to start", name_);
                on_close();
                try {
                    on_open();
                } catch (const std::exception& e) {
                    VP_LOG_ERROR("SourceNode '{}' loop reopen failed: {}", name_, e.what());
                    break;
                }
                frame = Frame();
                frame.stream_id = config_.stream_id;
                skip_counter = 0;
                continue;
            }

            VP_LOG_INFO("SourceNode '{}' reached end of stream at frame {}", name_, frame_counter);
            break;
        }

        ++skip_counter;
        if (config_.skip_frames > 0 && (skip_counter % (config_.skip_frames + 1)) != 1) {
            frame = Frame();
            frame.stream_id = config_.stream_id;
            continue;
        }

        frame.frame_id = frame_counter++;
        ++processed_count_;

        if (output_queue_) {
            output_queue_->push(std::move(frame));
        }

        frame = Frame();
        frame.stream_id = config_.stream_id;
    }

    on_close();

    state_ = NodeState::STOPPED;
    if (output_queue_ && !pipeline_managed_stop_) {
        output_queue_->stop();
    }
    VP_LOG_INFO("SourceNode '{}' stopped, total frames: {}", name_, frame_counter);
}

}  // namespace visionpipe
