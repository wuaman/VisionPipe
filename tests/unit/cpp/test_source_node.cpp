// 单元测试：SourceNode 抽象基类、Pipeline 合并拓扑、StreamError 异常、SourceConfig 扩展字段。
//
// 该文件仅包含黑盒测试，依赖头文件公开的接口。
// 通过 MockFileSource / MockSink 验证模板方法（on_open/read_next/on_close）
// 的调用时机与生命周期。

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/error.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/pipeline.h"
#include "core/source_node.h"
#include "nodes/source/source_config.h"

using namespace visionpipe;
using namespace std::chrono_literals;

namespace {

// ---------------------------------------------------------------------------
// MockFileSource: 仅用于验证 SourceNode 模板方法的最小实现
// ---------------------------------------------------------------------------
class MockFileSource : public SourceNode {
public:
    MockFileSource(const std::string& name, const SourceConfig& config, int max_frames)
        : SourceNode(name, config), max_frames_(max_frames) {}

    // 用于注入 on_open 的失败行为（再次打开时计数 +1，超过 fail_open_times_ 后成功）
    void set_fail_open_times(int times) { fail_open_times_ = times; }

    bool was_opened() const { return open_count_.load() > 0; }
    bool was_closed() const { return closed_.load(); }
    int open_count() const { return open_count_.load(); }
    int close_count() const { return close_count_.load(); }
    int produced_count() const { return produced_count_.load(); }

protected:
    void on_open() override {
        if (open_count_.load() < fail_open_times_) {
            open_count_.fetch_add(1);
            throw StreamError("MockFileSource: simulated open failure");
        }
        open_count_.fetch_add(1);
        opened_ = true;
        closed_ = false;
        // 每次重新打开都把帧计数归零（便于 loop / retry 测试）
        frames_emitted_ = 0;
    }

    bool read_next(Frame& frame) override {
        if (!opened_) {
            return false;
        }
        if (frames_emitted_ >= max_frames_) {
            return false;
        }
        frame.frame_id = static_cast<int64_t>(produced_count_.load());
        frame.stream_id = config_.stream_id;
        frame.pts_us = frames_emitted_ * 33333;
        ++frames_emitted_;
        produced_count_.fetch_add(1);
        // 模拟很轻量的解码耗时，避免循环空转干扰其他测试
        std::this_thread::sleep_for(2ms);
        return true;
    }

    void on_close() override {
        opened_ = false;
        closed_ = true;
        close_count_.fetch_add(1);
    }

private:
    int max_frames_;
    int frames_emitted_ = 0;
    std::atomic<int> fail_open_times_{0};
    std::atomic<int> open_count_{0};
    std::atomic<int> close_count_{0};
    std::atomic<int> produced_count_{0};
    std::atomic<bool> opened_{false};
    std::atomic<bool> closed_{false};
};

// ---------------------------------------------------------------------------
// MockSink: 用于 Pipeline 合并拓扑测试 - 收集帧到线程安全 vector
// ---------------------------------------------------------------------------
class MockSink : public NodeBase {
public:
    explicit MockSink(const std::string& name) : NodeBase(name) {}

    bool is_sink() const override { return true; }

    void process(Frame& frame) override {
        std::lock_guard<std::mutex> lock(mu_);
        stream_ids_.push_back(frame.stream_id);
        frame_count_.fetch_add(1);
    }

    size_t count() const { return frame_count_.load(); }

    std::unordered_set<int64_t> seen_stream_ids() const {
        std::lock_guard<std::mutex> lock(mu_);
        return std::unordered_set<int64_t>(stream_ids_.begin(), stream_ids_.end());
    }

    std::vector<int64_t> all_stream_ids() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stream_ids_;
    }

private:
    mutable std::mutex mu_;
    std::vector<int64_t> stream_ids_;
    std::atomic<size_t> frame_count_{0};
};

// 辅助函数：等待断言式条件最长 timeout
template <typename Pred>
bool wait_until(Pred pred, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    }
    return pred();
}

}  // namespace

// ===========================================================================
// 1. SourceConfig 新增字段的默认值与可独立修改性
// ===========================================================================
TEST(SourceConfigTest, NewFieldsHaveExpectedDefaults) {
    SourceConfig cfg;
    EXPECT_EQ(cfg.loop, false);
    EXPECT_EQ(cfg.skip_frames, 0);
    EXPECT_EQ(cfg.max_retries, 5);
    EXPECT_EQ(cfg.retry_interval_ms, 1000);
}

TEST(SourceConfigTest, NewFieldsIndependentlyMutable) {
    SourceConfig cfg;
    cfg.loop = true;
    cfg.skip_frames = 3;
    cfg.max_retries = 10;
    cfg.retry_interval_ms = 250;

    EXPECT_EQ(cfg.loop, true);
    EXPECT_EQ(cfg.skip_frames, 3);
    EXPECT_EQ(cfg.max_retries, 10);
    EXPECT_EQ(cfg.retry_interval_ms, 250);

    // 修改一个字段不影响其他字段
    cfg.loop = false;
    EXPECT_EQ(cfg.loop, false);
    EXPECT_EQ(cfg.skip_frames, 3);
    EXPECT_EQ(cfg.max_retries, 10);
    EXPECT_EQ(cfg.retry_interval_ms, 250);
}

// ===========================================================================
// 2. StreamError 异常
// ===========================================================================
TEST(StreamErrorTest, IsSubclassOfVisionPipeError) {
    try {
        throw StreamError("network down");
    } catch (const VisionPipeError& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("StreamError:"), std::string::npos);
        EXPECT_NE(msg.find("network down"), std::string::npos);
    }
}

TEST(StreamErrorTest, IsSubclassOfStdRuntimeError) {
    try {
        throw StreamError("decode failed");
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("StreamError:"), std::string::npos);
    }
}

TEST(StreamErrorTest, ThrowAndCatchAsStreamError) {
    EXPECT_THROW({ throw StreamError("oops"); }, StreamError);
}

TEST(StreamErrorTest, MessageContainsPrefix) {
    StreamError e("connection lost");
    std::string what(e.what());
    EXPECT_EQ(what.rfind("StreamError:", 0), 0u);
    EXPECT_NE(what.find("connection lost"), std::string::npos);
}

// ===========================================================================
// 3. SourceNode lifecycle
// ===========================================================================
TEST(SourceNodeLifecycleTest, ConstructionInitialState) {
    SourceConfig cfg;
    cfg.stream_id = 42;
    cfg.queue_capacity = 8;
    auto src = std::make_shared<MockFileSource>("src1", cfg, 3);

    EXPECT_EQ(src->state(), NodeState::INIT);
    EXPECT_EQ(src->is_source(), true);
    EXPECT_EQ(src->is_sink(), false);
    ASSERT_NE(src->output_queue(), nullptr);
    EXPECT_EQ(src->output_queue()->capacity(), 8u);
}

TEST(SourceNodeLifecycleTest, StartTransitionsToRunningAndProducesFrames) {
    SourceConfig cfg;
    cfg.stream_id = 7;
    cfg.queue_capacity = 32;
    auto src = std::make_shared<MockFileSource>("src", cfg, 5);

    ASSERT_NO_THROW(src->start());
    EXPECT_EQ(src->was_opened(), true);

    // 等待源自然结束（5 帧）
    src->wait_stop();

    EXPECT_EQ(src->produced_count(), 5);
    EXPECT_EQ(src->was_closed(), true);
    EXPECT_EQ(src->state(), NodeState::STOPPED);
}

TEST(SourceNodeLifecycleTest, OutputQueueStopsAtNaturalEof) {
    SourceConfig cfg;
    cfg.queue_capacity = 32;
    auto src = std::make_shared<MockFileSource>("src", cfg, 3);

    src->start();
    src->wait_stop();

    ASSERT_NE(src->output_queue(), nullptr);
    EXPECT_EQ(src->output_queue()->is_stopped(), true);
}

TEST(SourceNodeLifecycleTest, ExplicitStopCallsOnCloseAndTransitionsToStopped) {
    SourceConfig cfg;
    cfg.queue_capacity = 16;
    // 较多帧 + loop 防止过早 EOF，确保 stop() 真的中断 worker
    cfg.loop = true;
    auto src = std::make_shared<MockFileSource>("src", cfg, 5);

    src->start();
    // 让 worker 跑一会
    std::this_thread::sleep_for(50ms);
    src->stop();
    src->wait_stop();

    EXPECT_EQ(src->was_closed(), true);
    EXPECT_EQ(src->state(), NodeState::STOPPED);
}

TEST(SourceNodeLifecycleTest, WaitStopBlocksUntilThreadFinishes) {
    SourceConfig cfg;
    cfg.queue_capacity = 16;
    auto src = std::make_shared<MockFileSource>("src", cfg, 4);

    src->start();
    src->wait_stop();
    // wait_stop 返回后状态已经是 STOPPED
    EXPECT_EQ(src->state(), NodeState::STOPPED);
    EXPECT_EQ(src->produced_count(), 4);
}

TEST(SourceNodeLifecycleTest, StartIsIdempotent) {
    SourceConfig cfg;
    cfg.queue_capacity = 16;
    cfg.loop = true;  // 避免一开始就 EOF
    auto src = std::make_shared<MockFileSource>("src", cfg, 5);

    ASSERT_NO_THROW(src->start());
    // 再次调用不应崩溃
    ASSERT_NO_THROW(src->start());

    src->stop();
    src->wait_stop();
    EXPECT_EQ(src->state(), NodeState::STOPPED);
}

TEST(SourceNodeLifecycleTest, StartPropagatesOnOpenException) {
    SourceConfig cfg;
    cfg.queue_capacity = 16;
    // 让 on_open 失败 max_retries+1 次以保证 start 抛出
    cfg.max_retries = 0;
    auto src = std::make_shared<MockFileSource>("src", cfg, 3);
    src->set_fail_open_times(99);  // 始终失败

    EXPECT_THROW(src->start(), StreamError);
    // 失败后状态应保持 INIT 或转为 STOPPED（取决于实现，但至少不应 RUNNING）
    EXPECT_NE(src->state(), NodeState::RUNNING);
}

// ===========================================================================
// 4. SourceNode loop behavior
// ===========================================================================
TEST(SourceNodeLoopTest, LoopProducesMoreThanMaxFrames) {
    SourceConfig cfg;
    cfg.queue_capacity = 64;
    cfg.loop = true;
    auto src = std::make_shared<MockFileSource>("src", cfg, 5);

    src->start();
    // 让其循环 ~500ms（每帧 ~2ms → 应该 >> 5 帧）
    std::this_thread::sleep_for(500ms);
    src->stop();
    src->wait_stop();

    // loop 启用后应该跨越多个 epoch，远超 5 帧
    EXPECT_GT(src->produced_count(), 5);
    // 至少 on_open 被调用多次
    EXPECT_GT(src->open_count(), 1);
    EXPECT_EQ(src->state(), NodeState::STOPPED);
}

TEST(SourceNodeLoopTest, StopDuringLoopExits) {
    SourceConfig cfg;
    cfg.queue_capacity = 64;
    cfg.loop = true;
    auto src = std::make_shared<MockFileSource>("src", cfg, 5);

    src->start();
    std::this_thread::sleep_for(50ms);

    auto t0 = std::chrono::steady_clock::now();
    src->stop();
    src->wait_stop();
    auto elapsed = std::chrono::steady_clock::now() - t0;

    // stop 应在 1s 内退出
    EXPECT_LT(elapsed, 1s);
    EXPECT_EQ(src->state(), NodeState::STOPPED);
    EXPECT_EQ(src->was_closed(), true);
}

// ===========================================================================
// 5. SourceNode skip_frames behavior
// ===========================================================================
TEST(SourceNodeSkipFramesTest, SkipOneOutOfEveryTwo) {
    SourceConfig cfg;
    cfg.queue_capacity = 64;
    cfg.skip_frames = 1;  // 每 2 帧取 1 帧
    auto src = std::make_shared<MockFileSource>("src", cfg, 10);

    src->start();
    src->wait_stop();

    // 计数下游收到的帧数（由 output_queue 大小推断）
    auto stats = src->output_queue()->stats();
    // total_pushed 应当近似为 10 / 2 = 5
    EXPECT_EQ(stats.total_pushed, 5u);
    // read_next 实际被调用了 10 次（产出 10 帧再被采样过滤）
    EXPECT_EQ(src->produced_count(), 10);
}

TEST(SourceNodeSkipFramesTest, SkipTwoOutOfEveryThree) {
    SourceConfig cfg;
    cfg.queue_capacity = 64;
    cfg.skip_frames = 2;  // 每 3 帧取 1 帧
    auto src = std::make_shared<MockFileSource>("src", cfg, 9);

    src->start();
    src->wait_stop();

    auto stats = src->output_queue()->stats();
    // 9 / 3 = 3 帧
    EXPECT_EQ(stats.total_pushed, 3u);
    EXPECT_EQ(src->produced_count(), 9);
}

TEST(SourceNodeSkipFramesTest, SkipZeroEmitsAllFrames) {
    SourceConfig cfg;
    cfg.queue_capacity = 64;
    cfg.skip_frames = 0;
    auto src = std::make_shared<MockFileSource>("src", cfg, 6);

    src->start();
    src->wait_stop();

    auto stats = src->output_queue()->stats();
    EXPECT_EQ(stats.total_pushed, 6u);
}

// ===========================================================================
// 6. SourceNode retry behavior
// ===========================================================================
// read_next 一次返回 false 后,worker 会触发 on_close + on_open + retry。
// 用 loop=true 模拟 read_next 自然 EOF 后被 retry 路径处理。
TEST(SourceNodeRetryTest, LoopActsAsRetryRecovery) {
    SourceConfig cfg;
    cfg.queue_capacity = 32;
    cfg.loop = true;
    cfg.max_retries = 2;
    cfg.retry_interval_ms = 10;
    auto src = std::make_shared<MockFileSource>("src", cfg, 3);

    src->start();
    std::this_thread::sleep_for(200ms);
    src->stop();
    src->wait_stop();

    // 应当多次打开（loop） 并产生 > 3 帧
    EXPECT_GT(src->open_count(), 1);
    EXPECT_GT(src->produced_count(), 3);
}

// 当 max_retries=0 且非 loop 时,read_next 返回 false 后应直接结束(不重试)
TEST(SourceNodeRetryTest, NoRetryWhenMaxRetriesZeroAndNoLoop) {
    SourceConfig cfg;
    cfg.queue_capacity = 32;
    cfg.loop = false;
    cfg.max_retries = 0;
    auto src = std::make_shared<MockFileSource>("src", cfg, 3);

    src->start();
    src->wait_stop();

    // 只打开一次
    EXPECT_EQ(src->open_count(), 1);
    EXPECT_EQ(src->produced_count(), 3);
    EXPECT_EQ(src->state(), NodeState::STOPPED);
}

// ===========================================================================
// 7. Pipeline merge topology
// ===========================================================================
TEST(PipelineMergeTopologyTest, ThreeSourcesMergeIntoOneSink) {
    PipelineConfig pcfg;
    pcfg.name = "merge_test";
    pcfg.default_queue_capacity = 64;
    Pipeline pipeline(pcfg);

    SourceConfig s1cfg;
    s1cfg.stream_id = 101;
    s1cfg.queue_capacity = 32;
    auto s1 = std::make_shared<MockFileSource>("s1", s1cfg, 8);

    SourceConfig s2cfg;
    s2cfg.stream_id = 202;
    s2cfg.queue_capacity = 32;
    auto s2 = std::make_shared<MockFileSource>("s2", s2cfg, 8);

    SourceConfig s3cfg;
    s3cfg.stream_id = 303;
    s3cfg.queue_capacity = 32;
    auto s3 = std::make_shared<MockFileSource>("s3", s3cfg, 8);

    auto sink = std::make_shared<MockSink>("sink");

    pipeline.add_node(s1).add_node(s2).add_node(s3).add_node(sink);
    pipeline.connect(s1.get(), sink.get());
    pipeline.connect(s2.get(), sink.get());
    pipeline.connect(s3.get(), sink.get());

    pipeline.start();

    // 等待所有源消化完
    auto done = wait_until(
        [&]() { return sink->count() >= 24; },
        2s);
    pipeline.stop();
    pipeline.wait_stop();

    EXPECT_EQ(done, true);
    auto ids = sink->seen_stream_ids();
    EXPECT_EQ(ids.count(101), 1u);
    EXPECT_EQ(ids.count(202), 1u);
    EXPECT_EQ(ids.count(303), 1u);
    EXPECT_EQ(sink->count(), 24u);
}

// ===========================================================================
// 8. Pipeline merge stop behavior
// ===========================================================================
TEST(PipelineMergeStopTest, FastSourceFinishesWhileSlowSourceContinues) {
    PipelineConfig pcfg;
    pcfg.name = "merge_stop";
    pcfg.default_queue_capacity = 128;
    Pipeline pipeline(pcfg);

    SourceConfig fastcfg;
    fastcfg.stream_id = 1;
    fastcfg.queue_capacity = 64;
    auto fast = std::make_shared<MockFileSource>("fast", fastcfg, 5);

    SourceConfig slowcfg;
    slowcfg.stream_id = 2;
    slowcfg.queue_capacity = 64;
    auto slow = std::make_shared<MockFileSource>("slow", slowcfg, 50);

    auto sink = std::make_shared<MockSink>("sink");

    pipeline.add_node(fast).add_node(slow).add_node(sink);
    pipeline.connect(fast.get(), sink.get());
    pipeline.connect(slow.get(), sink.get());

    pipeline.start();

    // 等待两源都跑完
    auto all_done = wait_until(
        [&]() { return sink->count() >= 55; },
        3s);

    pipeline.stop();
    pipeline.wait_stop();

    EXPECT_EQ(all_done, true);
    EXPECT_EQ(sink->count(), 55u);
    EXPECT_EQ(fast->produced_count(), 5);
    EXPECT_EQ(slow->produced_count(), 50);

    // 共享 queue 最终应当被停止
    ASSERT_NE(fast->output_queue(), nullptr);
    EXPECT_EQ(fast->output_queue()->is_stopped(), true);
    // fast 和 slow 应共享同一个 output_queue（合并拓扑）
    EXPECT_EQ(fast->output_queue().get(), slow->output_queue().get());
}

TEST(PipelineMergeStopTest, StopExitsAllThreadsWithinOneSecond) {
    PipelineConfig pcfg;
    pcfg.name = "merge_stop_quick";
    pcfg.default_queue_capacity = 64;
    Pipeline pipeline(pcfg);

    SourceConfig c1;
    c1.stream_id = 1;
    c1.loop = true;
    c1.queue_capacity = 32;
    auto s1 = std::make_shared<MockFileSource>("s1", c1, 10);

    SourceConfig c2;
    c2.stream_id = 2;
    c2.loop = true;
    c2.queue_capacity = 32;
    auto s2 = std::make_shared<MockFileSource>("s2", c2, 10);

    auto sink = std::make_shared<MockSink>("sink");

    pipeline.add_node(s1).add_node(s2).add_node(sink);
    pipeline.connect(s1.get(), sink.get());
    pipeline.connect(s2.get(), sink.get());

    pipeline.start();
    std::this_thread::sleep_for(80ms);

    auto t0 = std::chrono::steady_clock::now();
    pipeline.stop();
    pipeline.wait_stop();
    auto elapsed = std::chrono::steady_clock::now() - t0;

    EXPECT_LT(elapsed, 1s);
    EXPECT_EQ(pipeline.state(), PipelineState::STOPPED);
}
