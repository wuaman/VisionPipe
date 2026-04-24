#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/error.h"
#include "core/node_base.h"
#include "core/pipeline.h"
#include "core/pipeline_manager.h"

using namespace std::chrono_literals;

namespace visionpipe {
namespace {

class MockSourceNode final : public NodeBase {
public:
    MockSourceNode(const std::string& name,
                   int64_t stream_id,
                   size_t total_frames,
                   std::chrono::milliseconds emit_delay = 0ms)
        : NodeBase(name)
        , stream_id_(stream_id)
        , total_frames_(total_frames)
        , emit_delay_(emit_delay) {}

    bool is_source() const override { return true; }

    void start() override {
        if (state_ == NodeState::RUNNING) {
            return;
        }
        state_ = NodeState::RUNNING;
        worker_thread_ = std::thread(&MockSourceNode::emit_frames, this);
    }

    void process(Frame&) override {}

    size_t emitted() const { return emitted_.load(); }

private:
    void emit_frames() {
        auto queue = output_queue();
        if (!queue) {
            state_ = NodeState::STOPPED;
            return;
        }

        for (size_t i = 0; i < total_frames_ && state_ == NodeState::RUNNING; ++i) {
            Frame frame;
            frame.stream_id = stream_id_;
            frame.frame_id = static_cast<int64_t>(i);
            frame.pts_us = static_cast<int64_t>(i) * 1000;
            queue->push(std::move(frame));
            emitted_.store(i + 1);

            if (emit_delay_.count() > 0) {
                std::this_thread::sleep_for(emit_delay_);
            }
        }

        state_ = NodeState::STOPPED;
        queue->stop();
    }

    int64_t stream_id_;
    size_t total_frames_;
    std::chrono::milliseconds emit_delay_;
    std::atomic<size_t> emitted_{0};
};

class MockSinkNode final : public NodeBase {
public:
    explicit MockSinkNode(const std::string& name,
                          std::chrono::milliseconds process_delay = 0ms)
        : NodeBase(name)
        , process_delay_(process_delay) {}

    bool is_sink() const override { return true; }

    void process(Frame& frame) override {
        if (process_delay_.count() > 0) {
            std::this_thread::sleep_for(process_delay_);
        }

        processed_.fetch_add(1);

        std::lock_guard<std::mutex> lock(mutex_);
        stream_ids_.push_back(frame.stream_id);
        frame_ids_.push_back(frame.frame_id);
    }

    size_t processed() const { return processed_.load(); }

    bool only_saw_stream(int64_t expected_stream_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !stream_ids_.empty() &&
               std::all_of(stream_ids_.begin(), stream_ids_.end(),
                           [expected_stream_id](int64_t value) { return value == expected_stream_id; });
    }

    std::vector<int64_t> frame_ids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frame_ids_;
    }

private:
    std::chrono::milliseconds process_delay_;
    std::atomic<size_t> processed_{0};

    mutable std::mutex mutex_;
    std::vector<int64_t> stream_ids_;
    std::vector<int64_t> frame_ids_;
};

struct MockPipelineBundle {
    std::string id;
    int64_t stream_id;
    size_t total_frames;
    PipelinePtr pipeline;
    std::shared_ptr<MockSourceNode> source;
    std::shared_ptr<MockSinkNode> sink;
};

bool WaitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds timeout = 2000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(1ms);
    }
    return predicate();
}

void ExpectSequentialFrameIds(const std::vector<int64_t>& ids, size_t expected_count) {
    ASSERT_EQ(ids.size(), expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        EXPECT_EQ(ids[i], static_cast<int64_t>(i));
    }
}

MockPipelineBundle MakeLinearPipeline(const std::string& name,
                                      const std::string& id,
                                      int64_t stream_id,
                                      size_t total_frames,
                                      size_t queue_capacity,
                                      std::chrono::milliseconds sink_delay,
                                      std::chrono::milliseconds source_delay = 0ms) {
    PipelineConfig config;
    config.name = name;
    config.id = id;
    config.default_queue_capacity = queue_capacity;
    config.default_overflow_policy = OverflowPolicy::BLOCK;

    auto pipeline = std::make_shared<Pipeline>(config);
    auto source = std::make_shared<MockSourceNode>(name + "_source", stream_id, total_frames, source_delay);
    auto sink = std::make_shared<MockSinkNode>(name + "_sink", sink_delay);

    source->create_output_queue(queue_capacity, OverflowPolicy::BLOCK);
    pipeline->add_node(source).add_node(sink).connect(source, sink);

    return MockPipelineBundle{
        .id = id,
        .stream_id = stream_id,
        .total_frames = total_frames,
        .pipeline = std::move(pipeline),
        .source = std::move(source),
        .sink = std::move(sink),
    };
}

TEST(PipelineManagerTest, CreateStatusListAndDestroySinglePipeline) {
    PipelineManager manager;

    EXPECT_TRUE(manager.list().empty());

    auto bundle = MakeLinearPipeline("alpha", "pipe-alpha", 101, 4, 8, 0ms);

    EXPECT_NO_THROW(manager.create(bundle.pipeline));
    EXPECT_EQ(manager.status(bundle.id), PipelineStatus::INIT);
    EXPECT_EQ(manager.list().size(), 1u);

    EXPECT_NO_THROW(manager.destroy(bundle.id));
    EXPECT_TRUE(manager.list().empty());
    EXPECT_THROW(manager.status(bundle.id), NotFoundError);
}

TEST(PipelineManagerTest, RejectsDuplicateCreateAndUnknownPipelineOperations) {
    PipelineManager manager;

    EXPECT_THROW(manager.start("missing"), NotFoundError);
    EXPECT_THROW(manager.stop("missing"), NotFoundError);
    EXPECT_THROW(manager.destroy("missing"), NotFoundError);
    EXPECT_THROW(manager.status("missing"), NotFoundError);

    auto first = MakeLinearPipeline("dup-a", "pipe-dup", 201, 2, 4, 0ms);
    auto duplicate = MakeLinearPipeline("dup-b", "pipe-dup", 202, 2, 4, 0ms);

    EXPECT_NO_THROW(manager.create(first.pipeline));
    EXPECT_THROW(manager.create(duplicate.pipeline), ConfigError);

    EXPECT_NO_THROW(manager.destroy(first.id));
    EXPECT_THROW(manager.destroy(first.id), NotFoundError);
}

TEST(PipelineManagerTest, StartsFivePipelinesIndependentlyAndStopsGracefully) {
    PipelineManager manager;
    std::vector<MockPipelineBundle> bundles;

    for (size_t i = 0; i < 5; ++i) {
        bundles.push_back(MakeLinearPipeline(
            "pipeline-" + std::to_string(i),
            "pipe-" + std::to_string(i),
            1000 + static_cast<int64_t>(i),
            10 + i,
            32,
            2ms));
    }

    for (auto& bundle : bundles) {
        EXPECT_NO_THROW(manager.create(bundle.pipeline));
    }
    EXPECT_EQ(manager.list().size(), 5u);

    for (auto& bundle : bundles) {
        EXPECT_NO_THROW(manager.start(bundle.id));
        ASSERT_TRUE(WaitFor([&manager, &bundle] {
            return manager.status(bundle.id) == PipelineStatus::RUNNING;
        })) << "pipeline did not reach RUNNING: " << bundle.id;
    }

    for (auto& bundle : bundles) {
        ASSERT_TRUE(WaitFor([&bundle] {
            return bundle.source->emitted() == bundle.total_frames;
        })) << "source did not emit all frames: " << bundle.id;
    }

    for (auto& bundle : bundles) {
        EXPECT_NO_THROW(manager.stop(bundle.id));
        EXPECT_EQ(manager.status(bundle.id), PipelineStatus::STOPPED);
        EXPECT_EQ(bundle.sink->processed(), bundle.total_frames);
        EXPECT_TRUE(bundle.sink->only_saw_stream(bundle.stream_id));
        ExpectSequentialFrameIds(bundle.sink->frame_ids(), bundle.total_frames);
    }

    for (auto& bundle : bundles) {
        EXPECT_NO_THROW(manager.destroy(bundle.id));
    }
    EXPECT_TRUE(manager.list().empty());
}

TEST(PipelineManagerTest, StopTransitionsThroughDrainingAndFlushesQueuedFrames) {
    PipelineManager manager;
    auto bundle = MakeLinearPipeline("drain", "pipe-drain", 303, 24, 32, 40ms);

    ASSERT_NO_THROW(manager.create(bundle.pipeline));
    ASSERT_NO_THROW(manager.start(bundle.id));
    ASSERT_TRUE(WaitFor([&manager, &bundle] {
        return manager.status(bundle.id) == PipelineStatus::RUNNING;
    })) << "pipeline did not reach RUNNING";

    ASSERT_TRUE(WaitFor([&bundle] {
        return bundle.source->emitted() == bundle.total_frames;
    })) << "source did not enqueue all frames before stop";

    ASSERT_LT(bundle.sink->processed(), bundle.total_frames);

    std::thread stopper([&manager, &bundle] { manager.stop(bundle.id); });

    EXPECT_TRUE(WaitFor([&manager, &bundle] {
        return manager.status(bundle.id) == PipelineStatus::DRAINING;
    }, 1000ms));

    stopper.join();

    EXPECT_EQ(manager.status(bundle.id), PipelineStatus::STOPPED);
    EXPECT_EQ(bundle.sink->processed(), bundle.total_frames);
    ExpectSequentialFrameIds(bundle.sink->frame_ids(), bundle.total_frames);

    EXPECT_NO_THROW(manager.destroy(bundle.id));
}

}  // namespace
}  // namespace visionpipe
