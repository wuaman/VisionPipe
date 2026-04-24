#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/infer_node.h"
#include "core/tensor.h"
#include "hal/imodel_engine.h"

namespace visionpipe {
namespace {

using namespace std::chrono_literals;

enum class DelayPattern {
    kConstant,
    kForceOutOfOrder,
};

class DelayedExecContext final : public IExecContext {
public:
    DelayedExecContext(std::chrono::milliseconds base_delay, DelayPattern pattern)
        : base_delay_(base_delay)
        , pattern_(pattern) {}

    void infer(const Tensor& input, Tensor& output) override {
        (void)output;

        if (input.data == nullptr || input.nbytes < sizeof(int32_t)) {
            throw std::runtime_error("test input tensor must contain a frame id");
        }

        const auto frame_id = *static_cast<const int32_t*>(input.data);
        std::this_thread::sleep_for(delay_for_frame(frame_id));
    }

private:
    std::chrono::milliseconds delay_for_frame(int32_t frame_id) const {
        if (pattern_ == DelayPattern::kConstant) {
            return base_delay_;
        }

        switch (frame_id % 3) {
            case 0:
                return base_delay_ * 6;
            case 1:
                return base_delay_;
            default:
                return base_delay_ * 3;
        }
    }

    std::chrono::milliseconds base_delay_;
    DelayPattern pattern_;
};

class DelayedModelEngine final : public IModelEngine {
public:
    DelayedModelEngine(std::chrono::milliseconds base_delay, DelayPattern pattern)
        : base_delay_(base_delay)
        , pattern_(pattern) {}

    std::unique_ptr<IExecContext> create_context() override {
        created_contexts_.fetch_add(1, std::memory_order_relaxed);
        return std::make_unique<DelayedExecContext>(base_delay_, pattern_);
    }

    size_t device_memory_bytes() const override { return 0; }

    size_t created_contexts() const { return created_contexts_.load(std::memory_order_relaxed); }

private:
    std::chrono::milliseconds base_delay_;
    DelayPattern pattern_;
    std::atomic<size_t> created_contexts_{0};
};

Frame make_frame(int64_t frame_id) {
    static CpuAllocator allocator;

    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = frame_id;
    frame.pts_us = frame_id * 1000;
    frame.image = Tensor({1}, DataType::INT32, &allocator);
    *static_cast<int32_t*>(frame.image.data) = static_cast<int32_t>(frame_id);
    return frame;
}

void expect_frame_ids_in_input_order(const std::vector<int64_t>& frame_ids, int64_t expected_count) {
    ASSERT_EQ(frame_ids.size(), static_cast<size_t>(expected_count));
    for (int64_t i = 0; i < expected_count; ++i) {
        EXPECT_EQ(frame_ids[static_cast<size_t>(i)], i) << "output frame order changed at index " << i;
    }
}

struct RunResult {
    std::vector<int64_t> frame_ids;
    double elapsed_seconds;
    size_t created_contexts;
    NodeState state_after_start;
    NodeState state_after_stop;
    NodeState state_after_wait;
};

RunResult run_infer_node(size_t workers,
                         int64_t frame_count,
                         std::chrono::milliseconds base_delay,
                         DelayPattern pattern) {
    auto engine = std::make_shared<DelayedModelEngine>(base_delay, pattern);
    InferNode node(engine, workers, "parallel-workers-test");
    BoundedQueue<Frame> input_queue(static_cast<size_t>(frame_count), OverflowPolicy::BLOCK);

    node.set_input_queue(&input_queue);
    node.create_output_queue(static_cast<size_t>(frame_count), OverflowPolicy::BLOCK);
    node.start();

    const auto state_after_start = node.state();
    const auto start_time = std::chrono::steady_clock::now();

    for (int64_t frame_id = 0; frame_id < frame_count; ++frame_id) {
        input_queue.push(make_frame(frame_id));
    }

    node.stop(true);
    const auto state_after_stop = node.state();
    input_queue.stop();
    node.wait_stop();

    const auto end_time = std::chrono::steady_clock::now();
    const auto state_after_wait = node.state();

    auto output_queue = node.output_queue();
    if (!output_queue) {
        throw std::runtime_error("InferNode did not create an output queue");
    }

    std::vector<int64_t> frame_ids;
    while (auto frame = output_queue->pop()) {
        frame_ids.push_back(frame->frame_id);
    }

    RunResult result;
    result.frame_ids = std::move(frame_ids);
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.created_contexts = engine->created_contexts();
    result.state_after_start = state_after_start;
    result.state_after_stop = state_after_stop;
    result.state_after_wait = state_after_wait;
    return result;
}

TEST(InferNodeParallelWorkersTest, SingleWorkerLifecycleDrainsAllFrames) {
    constexpr int64_t kFrameCount = 6;

    auto result = run_infer_node(1, kFrameCount, 20ms, DelayPattern::kConstant);

    EXPECT_EQ(result.created_contexts, 1u);
    EXPECT_EQ(result.state_after_start, NodeState::RUNNING);
    EXPECT_EQ(result.state_after_stop, NodeState::DRAINING);
    EXPECT_EQ(result.state_after_wait, NodeState::STOPPED);
    expect_frame_ids_in_input_order(result.frame_ids, kFrameCount);
}

TEST(InferNodeParallelWorkersTest, ThreeWorkersPreserveFrameOrderAfterOutOfOrderCompletion) {
    constexpr int64_t kFrameCount = 9;

    auto result = run_infer_node(3, kFrameCount, 10ms, DelayPattern::kForceOutOfOrder);

    EXPECT_EQ(result.created_contexts, 3u);
    EXPECT_EQ(result.state_after_start, NodeState::RUNNING);
    EXPECT_EQ(result.state_after_stop, NodeState::DRAINING);
    EXPECT_EQ(result.state_after_wait, NodeState::STOPPED);
    expect_frame_ids_in_input_order(result.frame_ids, kFrameCount);
}

TEST(InferNodeParallelWorkersTest, ThreeWorkersImproveThroughputByAtLeastTwoPointFiveX) {
    constexpr int64_t kFrameCount = 30;

    const auto single_worker = run_infer_node(1, kFrameCount, 40ms, DelayPattern::kConstant);
    const auto three_workers = run_infer_node(3, kFrameCount, 40ms, DelayPattern::kConstant);

    expect_frame_ids_in_input_order(single_worker.frame_ids, kFrameCount);
    expect_frame_ids_in_input_order(three_workers.frame_ids, kFrameCount);

    const double single_worker_throughput = static_cast<double>(kFrameCount) / single_worker.elapsed_seconds;
    const double three_worker_throughput = static_cast<double>(kFrameCount) / three_workers.elapsed_seconds;

    EXPECT_GE(three_worker_throughput, single_worker_throughput * 2.5)
        << "workers=1 throughput=" << single_worker_throughput
        << ", workers=3 throughput=" << three_worker_throughput;
}

}  // namespace
}  // namespace visionpipe
