// test_sink_node.cpp
// T4.4 单元测试：SinkNode 基类 enabled 机制

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "core/frame.h"
#include "core/node_base.h"
#include "core/tensor.h"
#include "nodes/sink/sink_node.h"
#include "nodes/sink/json_result_sink.h"
#include "nodes/sink/mjpeg_sink.h"

using namespace std::chrono_literals;

namespace visionpipe {
namespace {

static Frame make_rgb_frame(int h = 32, int w = 32,
                            int64_t frame_id = 0) {
    Frame f;
    f.frame_id = frame_id;
    f.stream_id = 1;
    f.pts_us = 1000;

    static CpuAllocator alloc;
    f.image = Tensor({h, w, 3}, DataType::UINT8, &alloc);
    std::memset(f.image.data, 128, f.image.nbytes);
    return f;
}

static void add_detection(Frame& f, int class_id, float conf) {
    Detection d;
    d.bbox[0] = 0.1f; d.bbox[1] = 0.2f;
    d.bbox[2] = 0.5f; d.bbox[3] = 0.6f;
    d.class_id = class_id;
    d.confidence = conf;
    f.detections.push_back(d);
}

// ============================================================================
// SinkNode 继承关系测试
// ============================================================================

TEST(SinkNodeHierarchyTest, JsonResultSinkIsSink) {
    JsonResultSink sink;
    EXPECT_TRUE(sink.is_sink());
    EXPECT_FALSE(sink.is_source());
}

TEST(SinkNodeHierarchyTest, MjpegSinkIsSink) {
    MjpegSink sink;
    EXPECT_TRUE(sink.is_sink());
    EXPECT_FALSE(sink.is_source());
}

// ============================================================================
// 默认 enabled 状态测试
// ============================================================================

TEST(SinkNodeDefaultEnabledTest, JsonResultSinkDefaultEnabled) {
    JsonResultSink sink;
    EXPECT_TRUE(sink.enabled());
}

TEST(SinkNodeDefaultEnabledTest, MjpegSinkDefaultDisabled) {
    MjpegSink sink;
    EXPECT_FALSE(sink.enabled());
}

// ============================================================================
// enabled 切换测试
// ============================================================================

class SinkNodeEnabledTest : public ::testing::Test {};

TEST_F(SinkNodeEnabledTest, SetEnabledTrue) {
    MjpegSink sink;
    EXPECT_FALSE(sink.enabled());
    sink.set_enabled(true);
    EXPECT_TRUE(sink.enabled());
}

TEST_F(SinkNodeEnabledTest, SetEnabledFalse) {
    JsonResultSink sink;
    EXPECT_TRUE(sink.enabled());
    sink.set_enabled(false);
    EXPECT_FALSE(sink.enabled());
}

TEST_F(SinkNodeEnabledTest, ToggleMultipleTimes) {
    JsonResultSink sink;
    for (int i = 0; i < 10; ++i) {
        sink.set_enabled(false);
        EXPECT_FALSE(sink.enabled());
        sink.set_enabled(true);
        EXPECT_TRUE(sink.enabled());
    }
}

// ============================================================================
// set_param 测试
// ============================================================================

class SinkNodeSetParamTest : public ::testing::Test {};

TEST_F(SinkNodeSetParamTest, EnabledParamSetToOne) {
    MjpegSink sink;
    EXPECT_FALSE(sink.enabled());
    bool ok = sink.set_param("enabled", 1);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(sink.enabled());
}

TEST_F(SinkNodeSetParamTest, EnabledParamSetToZero) {
    JsonResultSink sink;
    EXPECT_TRUE(sink.enabled());
    bool ok = sink.set_param("enabled", 0);
    EXPECT_TRUE(ok);
    EXPECT_FALSE(sink.enabled());
}

TEST_F(SinkNodeSetParamTest, UnknownParamStoredByBase) {
    JsonResultSink sink;
    bool ok = sink.set_param("nonexistent_param", 42);
    EXPECT_TRUE(ok);
}

TEST_F(SinkNodeSetParamTest, EnabledParamWrongTypeReturnsFalse) {
    JsonResultSink sink;
    bool ok = sink.set_param("enabled", std::string("true"));
    EXPECT_FALSE(ok);
    EXPECT_TRUE(sink.enabled());
}

TEST_F(SinkNodeSetParamTest, EnabledParamFloatTypeReturnsFalse) {
    JsonResultSink sink;
    bool ok = sink.set_param("enabled", 1.0f);
    EXPECT_FALSE(ok);
}

// ============================================================================
// enabled=false 时 process 跳过测试
// ============================================================================

TEST(SinkNodeSkipProcessTest, JsonResultSinkDisabled_NoOutput) {
    JsonResultSinkConfig cfg;
    JsonResultSink sink(cfg, "test_disabled");
    sink.set_enabled(false);

    Frame f = make_rgb_frame();
    add_detection(f, 1, 0.9f);
    sink.process(f);

    auto result = sink.pop_json(10ms);
    EXPECT_FALSE(result.has_value());
}

TEST(SinkNodeSkipProcessTest, JsonResultSinkEnabled_HasOutput) {
    JsonResultSinkConfig cfg;
    JsonResultSink sink(cfg, "test_enabled");
    EXPECT_TRUE(sink.enabled());

    Frame f = make_rgb_frame();
    add_detection(f, 1, 0.9f);
    sink.process(f);

    auto result = sink.pop_json(100ms);
    EXPECT_TRUE(result.has_value());
}

TEST(SinkNodeSkipProcessTest, MjpegSinkDefaultDisabled_NoOutput) {
    MjpegSink sink;
    EXPECT_FALSE(sink.enabled());

    Frame f = make_rgb_frame(64, 64);
    sink.process(f);

    auto result = sink.pop_jpeg(10ms);
    EXPECT_FALSE(result.has_value());
}

TEST(SinkNodeSkipProcessTest, MjpegSinkAfterEnable_HasOutput) {
    MjpegSink sink;
    sink.set_enabled(true);

    Frame f = make_rgb_frame(64, 64);
    sink.process(f);

    auto result = sink.pop_jpeg(100ms);
    EXPECT_TRUE(result.has_value());
    EXPECT_GT(result->size(), 0u);
}

// ============================================================================
// 运行时切换测试
// ============================================================================

TEST(SinkNodeRuntimeToggleTest, DisableThenReEnable_JsonSink) {
    JsonResultSink sink;

    // 初始 enabled，应有输出
    Frame f1 = make_rgb_frame(32, 32, 1);
    add_detection(f1, 0, 0.8f);
    sink.process(f1);
    auto r1 = sink.pop_json(100ms);
    EXPECT_TRUE(r1.has_value());

    // 禁用，不应有输出
    sink.set_enabled(false);
    Frame f2 = make_rgb_frame(32, 32, 2);
    add_detection(f2, 0, 0.8f);
    sink.process(f2);
    auto r2 = sink.pop_json(10ms);
    EXPECT_FALSE(r2.has_value());

    // 重新启用，应有输出
    sink.set_enabled(true);
    Frame f3 = make_rgb_frame(32, 32, 3);
    add_detection(f3, 0, 0.8f);
    sink.process(f3);
    auto r3 = sink.pop_json(100ms);
    EXPECT_TRUE(r3.has_value());
}

TEST(SinkNodeRuntimeToggleTest, EnableMjpegViaSinkParam) {
    MjpegSink sink;
    EXPECT_FALSE(sink.enabled());

    // 通过 set_param 启用
    sink.set_param("enabled", 1);
    EXPECT_TRUE(sink.enabled());

    Frame f = make_rgb_frame(64, 64);
    sink.process(f);

    auto result = sink.pop_jpeg(100ms);
    EXPECT_TRUE(result.has_value());
}

}  // namespace
}  // namespace visionpipe
