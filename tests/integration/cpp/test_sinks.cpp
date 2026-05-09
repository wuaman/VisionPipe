// test_sinks.cpp
// 任务 T4.4 集成测试：JsonResultSink 和 MjpegSink

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/tensor.h"
#include "nodes/sink/json_result_sink.h"
#include "nodes/sink/mjpeg_sink.h"

using namespace std::chrono_literals;

namespace visionpipe {
namespace {

// ============================================================================
// Test helpers
// ============================================================================

/// Build a minimal CPU RGB frame (HWC, UINT8).
static Frame make_rgb_frame(int h = 32, int w = 32,
                             int64_t frame_id = 0,
                             int64_t stream_id = 1,
                             int64_t pts_us = 1000) {
    Frame f;
    f.frame_id = frame_id;
    f.stream_id = stream_id;
    f.pts_us = pts_us;

    static CpuAllocator alloc;
    f.image = Tensor({h, w, 3}, DataType::UINT8, &alloc);
    // Fill with a solid colour so JPEG encode always produces valid output
    std::memset(f.image.data, 128, f.image.nbytes);
    return f;
}

/// Add a synthetic detection to a frame.
static void add_detection(Frame& f, int class_id, float conf, int64_t track_id = -1) {
    Detection d;
    d.bbox[0] = 0.1f;
    d.bbox[1] = 0.2f;
    d.bbox[2] = 0.5f;
    d.bbox[3] = 0.6f;
    d.class_id = class_id;
    d.confidence = conf;
    d.track_id = track_id;
    f.detections.push_back(d);
}

/// Add a synthetic track to a frame.
static void add_track(Frame& f, int64_t track_id, int class_id) {
    Track t;
    t.track_id = track_id;
    t.class_id = class_id;
    t.bbox[0] = 0.1f;
    t.bbox[1] = 0.2f;
    t.bbox[2] = 0.5f;
    t.bbox[3] = 0.6f;
    t.age = 5;
    t.confidence = 0.9f;
    f.tracks.push_back(t);
}

/// Run a node start/process/stop cycle for a single frame via worker thread.
static void run_single_frame(NodeBase& node, Frame frame) {
    BoundedQueue<Frame> q(4, OverflowPolicy::DROP_OLDEST);
    node.set_input_queue(&q);
    node.start();
    q.push(std::move(frame));
    std::this_thread::sleep_for(50ms);
    node.stop(/*drain=*/true);
    node.wait_stop();
}

// ============================================================================
// JsonResultSink Tests
// ============================================================================

class JsonResultSinkTest : public ::testing::Test {
protected:
    JsonResultSinkConfig cfg;
    std::unique_ptr<JsonResultSink> sink;

    void SetUp() override {
        cfg = JsonResultSinkConfig{};
        sink = std::make_unique<JsonResultSink>(cfg, "test_json_sink");
    }
};

TEST_F(JsonResultSinkTest, IsASinkNode) {
    EXPECT_TRUE(sink->is_sink());
    EXPECT_FALSE(sink->is_source());
}

TEST_F(JsonResultSinkTest, PopJsonTimeoutReturnsNulloptWhenEmpty) {
    auto result = sink->pop_json(10ms);
    EXPECT_FALSE(result.has_value());
}

TEST_F(JsonResultSinkTest, HappyPath_DetectionsSerializedCorrectly) {
    Frame f = make_rgb_frame(32, 32, /*frame_id=*/42, /*stream_id=*/7, /*pts_us=*/99999);
    add_detection(f, /*class_id=*/3, /*conf=*/0.95f, /*track_id=*/10);

    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());

    nlohmann::json j = nlohmann::json::parse(*json_opt);

    EXPECT_EQ(j["frame_id"].get<int64_t>(), 42);
    EXPECT_EQ(j["stream_id"].get<int64_t>(), 7);
    EXPECT_EQ(j["pts_us"].get<int64_t>(), 99999);

    ASSERT_TRUE(j.contains("detections"));
    ASSERT_EQ(j["detections"].size(), 1u);

    const auto& det = j["detections"][0];
    EXPECT_EQ(det["class_id"].get<int>(), 3);
    EXPECT_NEAR(det["confidence"].get<float>(), 0.95f, 1e-4f);
    EXPECT_EQ(det["track_id"].get<int64_t>(), 10);

    const auto& bbox = det["bbox"];
    ASSERT_EQ(bbox.size(), 4u);
    EXPECT_NEAR(bbox[0].get<float>(), 0.1f, 1e-4f);
    EXPECT_NEAR(bbox[1].get<float>(), 0.2f, 1e-4f);
    EXPECT_NEAR(bbox[2].get<float>(), 0.5f, 1e-4f);
    EXPECT_NEAR(bbox[3].get<float>(), 0.6f, 1e-4f);
}

TEST_F(JsonResultSinkTest, HappyPath_TracksSerializedCorrectly) {
    Frame f = make_rgb_frame();
    add_track(f, /*track_id=*/5, /*class_id=*/2);

    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());
    nlohmann::json j = nlohmann::json::parse(*json_opt);

    ASSERT_TRUE(j.contains("tracks"));
    ASSERT_EQ(j["tracks"].size(), 1u);

    const auto& tr = j["tracks"][0];
    EXPECT_EQ(tr["track_id"].get<int64_t>(), 5);
    EXPECT_EQ(tr["class_id"].get<int>(), 2);
    EXPECT_EQ(tr["age"].get<int>(), 5);
    EXPECT_NEAR(tr["confidence"].get<float>(), 0.9f, 1e-4f);
}

TEST_F(JsonResultSinkTest, EmptyFrame_ProducesValidJsonWithEmptyArrays) {
    Frame f;
    f.frame_id = 0;
    f.stream_id = 0;
    f.pts_us = 0;

    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());
    nlohmann::json j = nlohmann::json::parse(*json_opt);
    EXPECT_EQ(j["detections"].size(), 0u);
    EXPECT_EQ(j["tracks"].size(), 0u);
}

TEST_F(JsonResultSinkTest, MultipleDetections_AllSerialized) {
    Frame f = make_rgb_frame();
    for (int i = 0; i < 10; ++i) {
        add_detection(f, i, 0.5f + i * 0.01f);
    }

    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());
    nlohmann::json j = nlohmann::json::parse(*json_opt);
    EXPECT_EQ(j["detections"].size(), 10u);
}

TEST_F(JsonResultSinkTest, OutputIsParseable_Multiple_Frames) {
    for (int i = 0; i < 5; ++i) {
        Frame f = make_rgb_frame(32, 32, i);
        add_detection(f, 0, 0.9f);
        sink->process(f);
    }
    for (int i = 0; i < 5; ++i) {
        auto json_opt = sink->pop_json(100ms);
        ASSERT_TRUE(json_opt.has_value()) << "Missing frame " << i;
        EXPECT_NO_THROW(nlohmann::json::parse(*json_opt));
    }
}

TEST_F(JsonResultSinkTest, ConfigIncludeDetectionsFalse_NoDetectionsKey) {
    cfg.include_detections = false;
    sink = std::make_unique<JsonResultSink>(cfg, "test_json_sink2");

    Frame f = make_rgb_frame();
    add_detection(f, 0, 0.9f);
    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());
    nlohmann::json j = nlohmann::json::parse(*json_opt);
    EXPECT_FALSE(j.contains("detections"));
}

TEST_F(JsonResultSinkTest, ConfigIncludeTracksFalse_NoTracksKey) {
    cfg.include_tracks = false;
    sink = std::make_unique<JsonResultSink>(cfg, "test_json_sink3");

    Frame f = make_rgb_frame();
    add_track(f, 1, 0);
    sink->process(f);

    auto json_opt = sink->pop_json(100ms);
    ASSERT_TRUE(json_opt.has_value());
    nlohmann::json j = nlohmann::json::parse(*json_opt);
    EXPECT_FALSE(j.contains("tracks"));
}

TEST_F(JsonResultSinkTest, BufferOverflow_DropsOldest) {
    cfg.buffer_capacity = 3;
    sink = std::make_unique<JsonResultSink>(cfg, "test_json_overflow");

    // Push 5 frames — first 2 should be dropped (DROP_OLDEST)
    for (int64_t i = 0; i < 5; ++i) {
        Frame f = make_rgb_frame(32, 32, i);
        sink->process(f);
    }

    // Should get 3 frames, the most recent ones (IDs 2, 3, 4)
    int count = 0;
    while (auto opt = sink->pop_json(10ms)) {
        nlohmann::json j = nlohmann::json::parse(*opt);
        EXPECT_GE(j["frame_id"].get<int64_t>(), 2);
        ++count;
    }
    EXPECT_EQ(count, 3);
}

TEST_F(JsonResultSinkTest, ConcurrentPushPop_NoCrash) {
    std::atomic<int> pushed{0};
    std::atomic<int> popped{0};

    std::thread producer([&] {
        for (int i = 0; i < 20; ++i) {
            Frame f = make_rgb_frame(32, 32, i);
            add_detection(f, 0, 0.8f);
            sink->process(f);
            ++pushed;
            std::this_thread::sleep_for(1ms);
        }
    });

    std::thread consumer([&] {
        for (int i = 0; i < 30; ++i) {
            auto opt = sink->pop_json(20ms);
            if (opt.has_value()) {
                EXPECT_NO_THROW(nlohmann::json::parse(*opt));
                ++popped;
            }
        }
    });

    producer.join();
    consumer.join();
    EXPECT_GE(popped.load(), 1);
}

// ============================================================================
// MjpegSink Tests
// ============================================================================

class MjpegSinkTest : public ::testing::Test {
protected:
    MjpegSinkConfig cfg;
    std::unique_ptr<MjpegSink> sink;

    void SetUp() override {
        cfg = MjpegSinkConfig{};
        sink = std::make_unique<MjpegSink>(cfg, "test_mjpeg_sink");
    }

    static bool is_valid_jpeg(const std::vector<uint8_t>& buf) {
        if (buf.size() < 4) return false;
        // JPEG SOI marker
        if (buf[0] != 0xFF || buf[1] != 0xD8) return false;
        // JPEG EOI marker
        if (buf[buf.size() - 2] != 0xFF || buf[buf.size() - 1] != 0xD9) return false;
        return true;
    }
};

TEST_F(MjpegSinkTest, IsASinkNode) {
    EXPECT_TRUE(sink->is_sink());
    EXPECT_FALSE(sink->is_source());
}

TEST_F(MjpegSinkTest, PopJpegTimeoutReturnsNulloptWhenEmpty) {
    auto result = sink->pop_jpeg(10ms);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MjpegSinkTest, HappyPath_ProducesValidJpeg) {
    Frame f = make_rgb_frame(64, 64);
    sink->process(f);

    auto jpeg_opt = sink->pop_jpeg(100ms);
    ASSERT_TRUE(jpeg_opt.has_value());
    EXPECT_GT(jpeg_opt->size(), 0u);
    EXPECT_TRUE(is_valid_jpeg(*jpeg_opt));
}

TEST_F(MjpegSinkTest, JpegIsDecodeable_OpenCV) {
    Frame f = make_rgb_frame(64, 64);
    sink->process(f);

    auto jpeg_opt = sink->pop_jpeg(100ms);
    ASSERT_TRUE(jpeg_opt.has_value());

    cv::Mat decoded = cv::imdecode(*jpeg_opt, cv::IMREAD_COLOR);
    EXPECT_FALSE(decoded.empty());
    EXPECT_EQ(decoded.rows, 64);
    EXPECT_EQ(decoded.cols, 64);
}

TEST_F(MjpegSinkTest, NoImage_FrameSkipped) {
    Frame f;  // no image
    f.frame_id = 1;
    sink->process(f);

    auto result = sink->pop_jpeg(30ms);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MjpegSinkTest, LowerQuality_SmallerFileSize) {
    // Use random-noise image so JPEG quality has a visible size impact.
    static CpuAllocator alloc;
    const int H = 128, W = 128;
    Frame f1, f2;
    f1.image = Tensor({H, W, 3}, DataType::UINT8, &alloc);
    f2.image = Tensor({H, W, 3}, DataType::UINT8, &alloc);
    auto* p1 = static_cast<uint8_t*>(f1.image.data);
    auto* p2 = static_cast<uint8_t*>(f2.image.data);
    for (size_t i = 0; i < f1.image.nbytes; ++i) {
        p1[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
        p2[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);
    }

    MjpegSinkConfig high_cfg;
    high_cfg.jpeg_quality = 95;
    MjpegSink high_sink(high_cfg, "high_q");
    high_sink.process(f1);
    auto high_opt = high_sink.pop_jpeg(100ms);
    ASSERT_TRUE(high_opt.has_value());

    MjpegSinkConfig low_cfg;
    low_cfg.jpeg_quality = 10;
    MjpegSink low_sink(low_cfg, "low_q");
    low_sink.process(f2);
    auto low_opt = low_sink.pop_jpeg(100ms);
    ASSERT_TRUE(low_opt.has_value());

    EXPECT_LT(low_opt->size(), high_opt->size());
}

TEST_F(MjpegSinkTest, BufferOverflow_DropsOldest) {
    cfg.buffer_capacity = 2;
    sink = std::make_unique<MjpegSink>(cfg, "test_mjpeg_overflow");

    for (int i = 0; i < 5; ++i) {
        Frame f = make_rgb_frame(32, 32);
        sink->process(f);
    }

    int count = 0;
    while (auto opt = sink->pop_jpeg(10ms)) {
        EXPECT_TRUE(is_valid_jpeg(*opt));
        ++count;
    }
    EXPECT_EQ(count, 2);
}

TEST_F(MjpegSinkTest, MultipleFrames_AllValidJpeg) {
    cfg.buffer_capacity = 5;
    sink = std::make_unique<MjpegSink>(cfg, "test_mjpeg_multi");
    for (int i = 0; i < 5; ++i) {
        Frame f = make_rgb_frame(32, 32);
        sink->process(f);
    }
    int count = 0;
    while (auto opt = sink->pop_jpeg(10ms)) {
        EXPECT_TRUE(is_valid_jpeg(*opt));
        ++count;
    }
    EXPECT_EQ(count, 5);
}

TEST_F(MjpegSinkTest, ConcurrentPushPop_NoCrash) {
    std::atomic<int> valid_jpegs{0};

    std::thread producer([&] {
        for (int i = 0; i < 20; ++i) {
            Frame f = make_rgb_frame(32, 32);
            sink->process(f);
            std::this_thread::sleep_for(1ms);
        }
    });

    std::thread consumer([&] {
        for (int i = 0; i < 30; ++i) {
            auto opt = sink->pop_jpeg(20ms);
            if (opt.has_value()) {
                if (is_valid_jpeg(*opt)) ++valid_jpegs;
            }
        }
    });

    producer.join();
    consumer.join();
    EXPECT_GE(valid_jpegs.load(), 1);
}

}  // namespace
}  // namespace visionpipe
