#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "core/bounded_queue.h"
#include "core/error.h"
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

class TestBatchNode final : public InferNode {
public:
    TestBatchNode(std::shared_ptr<IModelEngine> engine,
                  size_t workers,
                  size_t max_batch_size = 1,
                  std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(5),
                  const std::string& name = "test-batch")
        : InferNode(std::move(engine), workers, max_batch_size, batch_timeout, name) {}

protected:
    void process_batch(std::vector<Frame>& frames) override {
        for (auto& frame : frames) {
            Tensor output;
            run_inference(frame.image, output);
        }
    }
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
                         DelayPattern pattern,
                         size_t max_batch_size = 1,
                         std::chrono::milliseconds batch_timeout = 5ms) {
    auto engine = std::make_shared<DelayedModelEngine>(base_delay, pattern);
    TestBatchNode node(engine, workers, max_batch_size, batch_timeout, "parallel-workers-test");
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

TEST(InferNodeParallelWorkersTest, BatchAccumulation) {
    constexpr int64_t kFrameCount = 12;

    auto result = run_infer_node(1, kFrameCount, 5ms, DelayPattern::kConstant,
                                 4, 50ms);

    expect_frame_ids_in_input_order(result.frame_ids, kFrameCount);
}

// -------------------------------------------------------------------------
// Additional batch-interface tests (process_batch / run_inference helpers)
// -------------------------------------------------------------------------

/// @brief A trivial mock context that records every infer / infer_multi call.
class RecordingExecContext final : public IExecContext {
public:
    explicit RecordingExecContext(std::atomic<size_t>* single_calls,
                                  std::atomic<size_t>* multi_calls)
        : single_calls_(single_calls), multi_calls_(multi_calls) {}

    void infer(const Tensor&, Tensor&) override {
        single_calls_->fetch_add(1, std::memory_order_relaxed);
    }

    void infer_multi(const Tensor&, std::vector<Tensor>& outputs) override {
        multi_calls_->fetch_add(1, std::memory_order_relaxed);
        outputs.clear();
    }

private:
    std::atomic<size_t>* single_calls_;
    std::atomic<size_t>* multi_calls_;
};

class RecordingModelEngine final : public IModelEngine {
public:
    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<RecordingExecContext>(&single_calls_, &multi_calls_);
    }

    size_t device_memory_bytes() const override { return 0; }

    size_t single_calls() const { return single_calls_.load(std::memory_order_relaxed); }
    size_t multi_calls() const { return multi_calls_.load(std::memory_order_relaxed); }

private:
    std::atomic<size_t> single_calls_{0};
    std::atomic<size_t> multi_calls_{0};
};

/// @brief Records batch sizes observed by process_batch.
class BatchSizeRecorderNode final : public InferNode {
public:
    BatchSizeRecorderNode(std::shared_ptr<IModelEngine> engine,
                          size_t workers,
                          size_t max_batch_size,
                          std::chrono::milliseconds batch_timeout)
        : InferNode(std::move(engine), workers, max_batch_size, batch_timeout, "recorder") {}

    std::vector<size_t> batch_sizes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return batch_sizes_;
    }

protected:
    void process_batch(std::vector<Frame>& frames) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch_sizes_.push_back(frames.size());
        }
        // Also exercise run_inference inside worker context (one call per frame).
        for (auto& frame : frames) {
            Tensor output;
            run_inference(frame.image, output);
        }
    }

private:
    mutable std::mutex mutex_;
    std::vector<size_t> batch_sizes_;
};

TEST(InferNodeBatchInterfaceTest, MaxBatchSizeIsRespectedWhenFramesAlreadyEnqueued) {
    constexpr size_t kMaxBatchSize = 4;
    constexpr int64_t kFrameCount = 12;  // multiple of kMaxBatchSize → all batches full

    auto engine = std::make_shared<RecordingModelEngine>();
    BatchSizeRecorderNode node(engine, /*workers=*/1, kMaxBatchSize, /*batch_timeout=*/200ms);

    BoundedQueue<Frame> input_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);
    node.set_input_queue(&input_queue);
    node.create_output_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);

    // Pre-fill the queue BEFORE start() so the worker sees a backlog and accumulates.
    for (int64_t frame_id = 0; frame_id < kFrameCount; ++frame_id) {
        input_queue.push(make_frame(frame_id));
    }

    node.start();
    node.stop(true);
    input_queue.stop();
    node.wait_stop();

    const auto sizes = node.batch_sizes();
    ASSERT_EQ(sizes.size(), static_cast<size_t>(kFrameCount) / kMaxBatchSize);
    for (size_t i = 0; i < sizes.size(); ++i) {
        EXPECT_EQ(sizes[i], kMaxBatchSize) << "batch index " << i << " was not full";
    }

    EXPECT_EQ(engine->single_calls(), static_cast<size_t>(kFrameCount));

    auto output_queue = node.output_queue();
    ASSERT_NE(output_queue, nullptr);
    std::vector<int64_t> ids;
    while (auto f = output_queue->pop()) {
        ids.push_back(f->frame_id);
    }
    expect_frame_ids_in_input_order(ids, kFrameCount);
}

TEST(InferNodeBatchInterfaceTest, PartialBatchIsFlushedAfterTimeout) {
    constexpr size_t kMaxBatchSize = 8;
    constexpr int64_t kFrameCount = 3;  // < max_batch_size → must rely on timeout
    constexpr auto kBatchTimeout = 30ms;

    auto engine = std::make_shared<RecordingModelEngine>();
    BatchSizeRecorderNode node(engine, /*workers=*/1, kMaxBatchSize, kBatchTimeout);

    BoundedQueue<Frame> input_queue(static_cast<size_t>(kMaxBatchSize), OverflowPolicy::BLOCK);
    node.set_input_queue(&input_queue);
    node.create_output_queue(static_cast<size_t>(kMaxBatchSize), OverflowPolicy::BLOCK);

    node.start();

    for (int64_t frame_id = 0; frame_id < kFrameCount; ++frame_id) {
        input_queue.push(make_frame(frame_id));
    }

    // Poll the output queue until all frames have flowed through. If the timeout
    // logic is broken the worker would wait forever for a full batch and this
    // poll loop times out.
    auto output_queue = node.output_queue();
    ASSERT_NE(output_queue, nullptr);

    std::vector<int64_t> ids;
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (ids.size() < static_cast<size_t>(kFrameCount)
           && std::chrono::steady_clock::now() < deadline) {
        if (auto frame = output_queue->pop_for(50ms)) {
            ids.push_back(frame->frame_id);
        }
    }

    node.stop(true);
    input_queue.stop();
    node.wait_stop();

    ASSERT_EQ(ids.size(), static_cast<size_t>(kFrameCount))
        << "partial batch was not flushed by timeout";
    expect_frame_ids_in_input_order(ids, kFrameCount);

    // All 3 frames were processed, none should have been dropped or duplicated.
    EXPECT_EQ(engine->single_calls(), static_cast<size_t>(kFrameCount));

    // The partial batch must be a single batch of size kFrameCount.
    const auto sizes = node.batch_sizes();
    ASSERT_EQ(sizes.size(), 1u);
    EXPECT_EQ(sizes[0], static_cast<size_t>(kFrameCount));
}

/// @brief Node that calls run_inference inside process_batch and exposes
///        the helper publicly so tests can invoke it from the main thread.
class HelperExposingNode final : public InferNode {
public:
    HelperExposingNode(std::shared_ptr<IModelEngine> engine,
                       size_t workers,
                       size_t max_batch_size,
                       std::chrono::milliseconds batch_timeout)
        : InferNode(std::move(engine), workers, max_batch_size, batch_timeout, "helper") {}

    // Re-export protected helpers for direct testing from outside worker context.
    using InferNode::run_inference;
    using InferNode::run_inference_multi;

protected:
    void process_batch(std::vector<Frame>& frames) override {
        for (auto& frame : frames) {
            Tensor single_output;
            run_inference(frame.image, single_output);

            std::vector<Tensor> multi_outputs;
            run_inference_multi(frame.image, multi_outputs);
        }
    }
};

TEST(InferNodeBatchInterfaceTest, RunInferenceHelpersInvokeContextInsideWorker) {
    constexpr int64_t kFrameCount = 5;
    constexpr size_t kMaxBatchSize = 2;

    auto engine = std::make_shared<RecordingModelEngine>();
    HelperExposingNode node(engine, /*workers=*/1, kMaxBatchSize, /*batch_timeout=*/20ms);

    BoundedQueue<Frame> input_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);
    node.set_input_queue(&input_queue);
    node.create_output_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);

    node.start();
    for (int64_t frame_id = 0; frame_id < kFrameCount; ++frame_id) {
        input_queue.push(make_frame(frame_id));
    }
    node.stop(true);
    input_queue.stop();
    node.wait_stop();

    // Each frame triggers exactly one single-output and one multi-output call.
    EXPECT_EQ(engine->single_calls(), static_cast<size_t>(kFrameCount));
    EXPECT_EQ(engine->multi_calls(), static_cast<size_t>(kFrameCount));
}

TEST(InferNodeBatchInterfaceTest, RunInferenceThrowsInferErrorOutsideWorkerContext) {
    auto engine = std::make_shared<RecordingModelEngine>();
    HelperExposingNode node(engine, /*workers=*/1, /*max_batch_size=*/1, /*batch_timeout=*/5ms);

    // Node not started → no worker thread → thread-local current_context_ is null.
    static CpuAllocator allocator;
    Tensor input({1}, DataType::INT32, &allocator);
    *static_cast<int32_t*>(input.data) = 0;
    Tensor output;

    EXPECT_THROW(node.run_inference(input, output), InferError);

    // Helper must remain unchanged: zero infer calls reached the engine.
    EXPECT_EQ(engine->single_calls(), 0u);
}

TEST(InferNodeBatchInterfaceTest, RunInferenceMultiThrowsInferErrorOutsideWorkerContext) {
    auto engine = std::make_shared<RecordingModelEngine>();
    HelperExposingNode node(engine, /*workers=*/1, /*max_batch_size=*/1, /*batch_timeout=*/5ms);

    static CpuAllocator allocator;
    Tensor input({1}, DataType::INT32, &allocator);
    *static_cast<int32_t*>(input.data) = 0;
    std::vector<Tensor> outputs;

    EXPECT_THROW(node.run_inference_multi(input, outputs), InferError);

    EXPECT_EQ(engine->multi_calls(), 0u);
}

/// @brief Node whose process_batch unconditionally throws — used to test
///        per-batch error accounting.
class ThrowingBatchNode final : public InferNode {
public:
    ThrowingBatchNode(std::shared_ptr<IModelEngine> engine,
                      size_t workers,
                      size_t max_batch_size,
                      std::chrono::milliseconds batch_timeout)
        : InferNode(std::move(engine), workers, max_batch_size, batch_timeout, "throwing") {}

    size_t process_batch_calls() const {
        return process_batch_calls_.load(std::memory_order_relaxed);
    }

protected:
    void process_batch(std::vector<Frame>&) override {
        process_batch_calls_.fetch_add(1, std::memory_order_relaxed);
        throw InferError("synthetic batch failure");
    }

private:
    std::atomic<size_t> process_batch_calls_{0};
};

TEST(InferNodeBatchInterfaceTest, ProcessBatchThrowCountsEveryFrameAsError) {
    constexpr size_t kMaxBatchSize = 4;
    constexpr int64_t kFrameCount = 8;  // exactly two full batches

    auto engine = std::make_shared<RecordingModelEngine>();
    ThrowingBatchNode node(engine, /*workers=*/1, kMaxBatchSize, /*batch_timeout=*/200ms);

    BoundedQueue<Frame> input_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);
    node.set_input_queue(&input_queue);
    node.create_output_queue(static_cast<size_t>(kFrameCount), OverflowPolicy::BLOCK);

    // Pre-fill so both batches are full when the worker starts.
    for (int64_t frame_id = 0; frame_id < kFrameCount; ++frame_id) {
        input_queue.push(make_frame(frame_id));
    }

    node.start();
    node.stop(true);
    input_queue.stop();
    node.wait_stop();

    // Two batches of 4 → exactly 2 process_batch invocations.
    EXPECT_EQ(node.process_batch_calls(), 2u);

    const auto stats = node.stats();
    EXPECT_EQ(stats.processed_count, 0u);
    EXPECT_EQ(stats.error_count, static_cast<uint64_t>(kFrameCount));

    // No frames should have escaped to the downstream queue.
    auto output_queue = node.output_queue();
    ASSERT_NE(output_queue, nullptr);
    EXPECT_EQ(output_queue->size(), 0u);
}

}  // namespace
}  // namespace visionpipe
