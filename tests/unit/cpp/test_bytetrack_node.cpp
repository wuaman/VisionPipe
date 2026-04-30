// test_bytetrack_node.cpp
// 任务 T2.5 单元测试：ByteTrackNode + ByteTrackImpl

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "nodes/tracker/bytetrack_impl.h"
#include "nodes/tracker/bytetrack_node.h"

namespace visionpipe {
namespace {

using namespace std::chrono_literals;

// ==================== TrackBox 测试 ====================

class TrackBoxTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TrackBoxTest, DefaultValues) {
    TrackBox box;

    EXPECT_FLOAT_EQ(box.bbox[0], 0.0f);
    EXPECT_FLOAT_EQ(box.bbox[1], 0.0f);
    EXPECT_FLOAT_EQ(box.bbox[2], 0.0f);
    EXPECT_FLOAT_EQ(box.bbox[3], 0.0f);
    EXPECT_EQ(box.class_id, 0);
    EXPECT_FLOAT_EQ(box.confidence, 0.0f);
    EXPECT_EQ(box.track_id, -1);
}

TEST_F(TrackBoxTest, GeometryMethods) {
    TrackBox box;
    box.bbox[0] = 0.1f;
    box.bbox[1] = 0.2f;
    box.bbox[2] = 0.5f;
    box.bbox[3] = 0.6f;

    EXPECT_FLOAT_EQ(box.cx(), 0.3f);
    EXPECT_FLOAT_EQ(box.cy(), 0.4f);
    EXPECT_FLOAT_EQ(box.width(), 0.4f);
    EXPECT_FLOAT_EQ(box.height(), 0.4f);
    EXPECT_FLOAT_EQ(box.area(), 0.16f);
}

TEST_F(TrackBoxTest, IoUSameBox) {
    TrackBox box;
    box.bbox[0] = 0.0f;
    box.bbox[1] = 0.0f;
    box.bbox[2] = 1.0f;
    box.bbox[3] = 1.0f;

    float iou = box.iou(box);
    EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST_F(TrackBoxTest, IoUNoOverlap) {
    TrackBox box1;
    box1.bbox[0] = 0.0f;
    box1.bbox[1] = 0.0f;
    box1.bbox[2] = 0.5f;
    box1.bbox[3] = 0.5f;

    TrackBox box2;
    box2.bbox[0] = 0.6f;
    box2.bbox[1] = 0.6f;
    box2.bbox[2] = 1.0f;
    box2.bbox[3] = 1.0f;

    float iou = box1.iou(box2);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST_F(TrackBoxTest, IoUPartialOverlap) {
    TrackBox box1;
    box1.bbox[0] = 0.0f;
    box1.bbox[1] = 0.0f;
    box1.bbox[2] = 0.6f;
    box1.bbox[3] = 0.6f;

    TrackBox box2;
    box2.bbox[0] = 0.4f;
    box2.bbox[1] = 0.4f;
    box2.bbox[2] = 1.0f;
    box2.bbox[3] = 1.0f;

    float iou = box1.iou(box2);
    EXPECT_GT(iou, 0.0f);
    EXPECT_LT(iou, 1.0f);
}

TEST_F(TrackBoxTest, IoUZeroArea) {
    TrackBox box1;
    box1.bbox[0] = 0.5f;
    box1.bbox[1] = 0.5f;
    box1.bbox[2] = 0.5f;
    box1.bbox[3] = 0.5f;

    TrackBox box2;
    box2.bbox[0] = 0.0f;
    box2.bbox[1] = 0.0f;
    box2.bbox[2] = 1.0f;
    box2.bbox[3] = 1.0f;

    float iou = box1.iou(box2);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

// ==================== TrackedObject 测试 ====================

class TrackedObjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        box_.bbox[0] = 0.1f;
        box_.bbox[1] = 0.2f;
        box_.bbox[2] = 0.3f;
        box_.bbox[3] = 0.4f;
        box_.class_id = 1;
        box_.confidence = 0.9f;
    }

    TrackBox box_;
};

TEST_F(TrackedObjectTest, Construction) {
    TrackedObject trk(42, box_);

    EXPECT_EQ(trk.id(), 42);
    EXPECT_EQ(trk.state(), TrackedObject::State::New);
    EXPECT_EQ(trk.class_id(), 1);
    EXPECT_FLOAT_EQ(trk.confidence(), 0.9f);
    EXPECT_EQ(trk.age(), 1);
    EXPECT_EQ(trk.hit_streak(), 0);
    EXPECT_EQ(trk.miss_count(), 0);
}

TEST_F(TrackedObjectTest, Update) {
    TrackedObject trk(1, box_);

    TrackBox new_box;
    new_box.bbox[0] = 0.15f;
    new_box.bbox[1] = 0.25f;
    new_box.bbox[2] = 0.35f;
    new_box.bbox[3] = 0.45f;
    new_box.confidence = 0.85f;

    trk.update(new_box);

    EXPECT_EQ(trk.state(), TrackedObject::State::Tracked);
    EXPECT_EQ(trk.hit_streak(), 1);
    EXPECT_EQ(trk.miss_count(), 0);
    EXPECT_FLOAT_EQ(trk.confidence(), 0.85f);
}

TEST_F(TrackedObjectTest, MarkLost) {
    TrackedObject trk(1, box_);
    trk.update(box_);
    trk.mark_lost();

    EXPECT_EQ(trk.state(), TrackedObject::State::Lost);
    EXPECT_EQ(trk.hit_streak(), 0);
    EXPECT_EQ(trk.miss_count(), 1);
}

TEST_F(TrackedObjectTest, MarkRemoved) {
    TrackedObject trk(1, box_);
    trk.mark_removed();

    EXPECT_EQ(trk.state(), TrackedObject::State::Removed);
}

TEST_F(TrackedObjectTest, PredictIncreasesAge) {
    TrackedObject trk(1, box_);

    trk.predict();

    EXPECT_EQ(trk.age(), 2);
}

TEST_F(TrackedObjectTest, MultipleUpdates) {
    TrackedObject trk(1, box_);

    for (int i = 0; i < 10; ++i) {
        TrackBox new_box = box_;
        new_box.bbox[0] += 0.01f * i;
        trk.update(new_box);
    }

    EXPECT_EQ(trk.state(), TrackedObject::State::Tracked);
    EXPECT_EQ(trk.hit_streak(), 10);
    EXPECT_EQ(trk.miss_count(), 0);
}

// ==================== ByteTrackImpl 测试 ====================

class ByteTrackImplTest : public ::testing::Test {
protected:
    void SetUp() override {
        tracker_ = std::make_unique<ByteTrackImpl>(0.5f, 30, 0.3f, 30);
    }

    void TearDown() override {}

    TrackBox make_detection(float x1, float y1, float x2, float y2,
                            int class_id = 0, float conf = 0.9f) {
        TrackBox box;
        box.bbox[0] = x1;
        box.bbox[1] = y1;
        box.bbox[2] = x2;
        box.bbox[3] = y2;
        box.class_id = class_id;
        box.confidence = conf;
        return box;
    }

    std::unique_ptr<ByteTrackImpl> tracker_;
};

TEST_F(ByteTrackImplTest, EmptyDetections) {
    std::vector<TrackBox> detections;
    auto tracks = tracker_->update(detections);

    EXPECT_TRUE(tracks.empty());
}

TEST_F(ByteTrackImplTest, SingleDetectionCreatesTrack) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    auto tracks = tracker_->update(detections);

    // 新轨迹需要 hit_streak >= 1 才输出
    EXPECT_TRUE(tracks.empty());

    // 再次更新同一检测
    tracks = tracker_->update(detections);

    // 现在应该有输出
    EXPECT_EQ(tracks.size(), 1u);
}

TEST_F(ByteTrackImplTest, TrackPersistsAcrossFrames) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    // 第一帧：创建轨迹
    tracker_->update(detections);

    // 第二帧：确认轨迹
    auto tracks = tracker_->update(detections);
    EXPECT_EQ(tracks.size(), 1u);

    int64_t first_track_id = tracks[0].id();

    // 后续帧：轨迹保持
    for (int i = 0; i < 5; ++i) {
        tracks = tracker_->update(detections);
        EXPECT_EQ(tracks.size(), 1u);
        EXPECT_EQ(tracks[0].id(), first_track_id);
    }
}

TEST_F(ByteTrackImplTest, TrackLostAfterNoDetections) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    // 创建并确认轨迹
    tracker_->update(detections);
    auto tracks = tracker_->update(detections);
    EXPECT_EQ(tracks.size(), 1u);

    // 无检测帧
    std::vector<TrackBox> empty_dets;
    for (int i = 0; i < 10; ++i) {
        tracker_->update(empty_dets);
    }

    // 轨迹应该丢失
    auto& all_tracks = tracker_->tracks();
    EXPECT_TRUE(all_tracks.empty() || all_tracks[0].state() == TrackedObject::State::Lost);
}

TEST_F(ByteTrackImplTest, MultipleDetectionsCreateMultipleTracks) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.2f, 0.2f, 0, 0.9f));
    detections.push_back(make_detection(0.5f, 0.5f, 0.7f, 0.7f, 0, 0.9f));
    detections.push_back(make_detection(0.8f, 0.1f, 0.9f, 0.3f, 1, 0.8f));

    // 第一帧
    tracker_->update(detections);

    // 第二帧
    auto tracks = tracker_->update(detections);

    EXPECT_EQ(tracks.size(), 3u);
}

TEST_F(ByteTrackImplTest, DifferentClassesNotMerged) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));
    detections.push_back(make_detection(0.15f, 0.15f, 0.35f, 0.35f, 1, 0.9f));

    tracker_->update(detections);
    auto tracks = tracker_->update(detections);

    // 不同类别的检测应创建不同轨迹
    EXPECT_EQ(tracks.size(), 2u);
}

TEST_F(ByteTrackImplTest, LowConfidenceDetections) {
    std::vector<TrackBox> high_dets;
    high_dets.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    std::vector<TrackBox> low_dets;
    low_dets.push_back(make_detection(0.15f, 0.15f, 0.35f, 0.35f, 0, 0.4f));

    // 高置信度检测创建轨迹
    tracker_->update(high_dets);
    auto tracks = tracker_->update(high_dets);
    EXPECT_EQ(tracks.size(), 1u);

    // 低置信度检测不应创建新轨迹
    tracker_->update(low_dets);
    EXPECT_LE(tracker_->tracks().size(), 1u);
}

TEST_F(ByteTrackImplTest, Reset) {
    std::vector<TrackBox> detections;
    detections.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    tracker_->update(detections);
    tracker_->update(detections);
    EXPECT_FALSE(tracker_->tracks().empty());

    tracker_->reset();
    EXPECT_TRUE(tracker_->tracks().empty());
}

TEST_F(ByteTrackImplTest, TrackIdUniqueness) {
    std::vector<TrackBox> detections1;
    detections1.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));

    tracker_->update(detections1);
    auto tracks1 = tracker_->update(detections1);
    int64_t id1 = tracks1[0].id();

    // 添加第二个检测，应获得不同 ID
    std::vector<TrackBox> detections2;
    detections2.push_back(make_detection(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));
    detections2.push_back(make_detection(0.5f, 0.5f, 0.7f, 0.7f, 0, 0.9f));

    tracker_->update(detections2);
    auto tracks2 = tracker_->update(detections2);

    // 应有两个不同 ID 的轨迹
    EXPECT_EQ(tracks2.size(), 2u);
    if (tracks2.size() >= 2) {
        EXPECT_NE(tracks2[0].id(), tracks2[1].id());
    }
}

// ==================== ByteTrackConfig 测试 ====================

class ByteTrackConfigTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ByteTrackConfigTest, DefaultValues) {
    ByteTrackConfig config;

    EXPECT_FLOAT_EQ(config.track_thresh, 0.5f);
    EXPECT_EQ(config.track_buffer, 30);
    EXPECT_FLOAT_EQ(config.match_thresh, 0.3f);
    EXPECT_EQ(config.frame_rate, 30);
}

TEST_F(ByteTrackConfigTest, CustomValues) {
    ByteTrackConfig config;
    config.track_thresh = 0.7f;
    config.track_buffer = 60;
    config.match_thresh = 0.5f;
    config.frame_rate = 25;

    EXPECT_FLOAT_EQ(config.track_thresh, 0.7f);
    EXPECT_EQ(config.track_buffer, 60);
    EXPECT_FLOAT_EQ(config.match_thresh, 0.5f);
    EXPECT_EQ(config.frame_rate, 25);
}

// ==================== ByteTrackNode 构造测试 ====================

class ByteTrackNodeConstructorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ByteTrackNodeConstructorTest, DefaultConstruction) {
    ByteTrackNode node;

    EXPECT_EQ(node.config().track_thresh, 0.5f);
    EXPECT_EQ(node.config().track_buffer, 30);
    EXPECT_EQ(node.state(), NodeState::INIT);
}

TEST_F(ByteTrackNodeConstructorTest, ConstructionWithConfig) {
    ByteTrackConfig config;
    config.track_thresh = 0.7f;
    config.track_buffer = 45;
    config.match_thresh = 0.4f;
    config.frame_rate = 25;

    ByteTrackNode node(config, "custom_tracker");

    EXPECT_EQ(node.config().track_thresh, 0.7f);
    EXPECT_EQ(node.config().track_buffer, 45);
    EXPECT_EQ(node.config().match_thresh, 0.4f);
    EXPECT_EQ(node.config().frame_rate, 25);
    EXPECT_EQ(node.name(), "custom_tracker");
}

TEST_F(ByteTrackNodeConstructorTest, CannotMoveDueToUniquePtr) {
    // ByteTrackNode contains unique_ptr member, move is deleted
    // Typically managed via shared_ptr
    auto node_ptr = std::make_shared<ByteTrackNode>(ByteTrackConfig{}, "original_tracker");
    EXPECT_EQ(node_ptr->name(), "original_tracker");
}

// ==================== ByteTrackNode 参数测试 ====================

class ByteTrackNodeParamTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_unique<ByteTrackNode>();
    }

    void TearDown() override {}

    std::unique_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeParamTest, SetTrackThresh) {
    bool result = node_->set_param("track_thresh", 0.6f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().track_thresh, 0.6f);
}

TEST_F(ByteTrackNodeParamTest, SetTrackBuffer) {
    bool result = node_->set_param("track_buffer", 50);
    EXPECT_TRUE(result);
    EXPECT_EQ(node_->config().track_buffer, 50);
}

TEST_F(ByteTrackNodeParamTest, SetMatchThresh) {
    bool result = node_->set_param("match_thresh", 0.5f);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().match_thresh, 0.5f);
}

TEST_F(ByteTrackNodeParamTest, SetFrameRate) {
    bool result = node_->set_param("frame_rate", 60);
    EXPECT_TRUE(result);
    EXPECT_EQ(node_->config().frame_rate, 60);
}

TEST_F(ByteTrackNodeParamTest, SetInvalidParamName) {
    bool result = node_->set_param("invalid_param", 123);
    EXPECT_FALSE(result);
}

TEST_F(ByteTrackNodeParamTest, SetParamWithDouble) {
    // 测试 double 转 float
    bool result = node_->set_param("track_thresh", 0.65);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(node_->config().track_thresh, 0.65f);
}

// ==================== ByteTrackNode 处理测试 ====================

Frame make_tracking_frame(int64_t frame_id, const std::vector<Detection>& dets) {
    Frame frame;
    frame.stream_id = 1;
    frame.frame_id = frame_id;
    frame.pts_us = frame_id * 33333;
    frame.detections = dets;
    return frame;
}

Detection make_detection_for_track(float x1, float y1, float x2, float y2,
                                   int class_id = 0, float conf = 0.9f) {
    Detection det;
    det.bbox[0] = x1;
    det.bbox[1] = y1;
    det.bbox[2] = x2;
    det.bbox[3] = y2;
    det.class_id = class_id;
    det.confidence = conf;
    det.track_id = -1;
    return det;
}

class ByteTrackNodeProcessTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_unique<ByteTrackNode>();
    }

    void TearDown() override {}

    std::unique_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeProcessTest, ProcessEmptyDetections) {
    Frame frame = make_tracking_frame(0, {});

    node_->process(frame);

    EXPECT_TRUE(frame.detections.empty());
    EXPECT_TRUE(frame.tracks.empty());
}

TEST_F(ByteTrackNodeProcessTest, ProcessSingleDetection) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    Frame frame = make_tracking_frame(0, dets);

    node_->process(frame);

    // 第一次处理可能没有输出轨迹
    EXPECT_GE(frame.detections.size(), 1u);

    // 再次处理
    Frame frame2 = make_tracking_frame(1, dets);
    node_->process(frame2);

    EXPECT_EQ(frame2.detections.size(), 1u);
    EXPECT_GE(frame2.detections[0].track_id, 0);
}

TEST_F(ByteTrackNodeProcessTest, TrackIdConsistentAcrossFrames) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    // 第一帧
    Frame frame1 = make_tracking_frame(0, dets);
    node_->process(frame1);

    // 第二帧
    Frame frame2 = make_tracking_frame(1, dets);
    node_->process(frame2);

    int64_t track_id = frame2.detections[0].track_id;

    // 后续帧应保持相同 track_id
    for (int i = 2; i < 10; ++i) {
        Frame frame = make_tracking_frame(i, dets);
        node_->process(frame);

        if (!frame.detections.empty()) {
            EXPECT_EQ(frame.detections[0].track_id, track_id);
        }
    }
}

TEST_F(ByteTrackNodeProcessTest, MultipleDetectionsGetDifferentTrackIds) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.9f));
    dets.push_back(make_detection_for_track(0.5f, 0.5f, 0.7f, 0.7f, 0, 0.9f));

    // 第一帧
    Frame frame1 = make_tracking_frame(0, dets);
    node_->process(frame1);

    // 第二帧
    Frame frame2 = make_tracking_frame(1, dets);
    node_->process(frame2);

    EXPECT_EQ(frame2.detections.size(), 2u);

    // 两个检测应有不同的 track_id
    if (frame2.detections.size() >= 2) {
        EXPECT_NE(frame2.detections[0].track_id, frame2.detections[1].track_id);
    }
}

TEST_F(ByteTrackNodeProcessTest, TracksVectorPopulated) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    // 第一帧
    Frame frame1 = make_tracking_frame(0, dets);
    node_->process(frame1);

    // 第二帧
    Frame frame2 = make_tracking_frame(1, dets);
    node_->process(frame2);

    // tracks 向量应有内容
    EXPECT_GE(frame2.tracks.size(), 1u);

    if (!frame2.tracks.empty()) {
        const Track& track = frame2.tracks[0];
        EXPECT_GE(track.track_id, 0);
        EXPECT_EQ(track.class_id, 0);
        EXPECT_GT(track.age, 0);
    }
}

TEST_F(ByteTrackNodeProcessTest, ResetClearsTracks) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    Frame frame1 = make_tracking_frame(0, dets);
    node_->process(frame1);

    Frame frame2 = make_tracking_frame(1, dets);
    node_->process(frame2);
    EXPECT_GE(node_->active_track_count(), 1u);

    node_->reset();

    EXPECT_EQ(node_->active_track_count(), 0u);
}

// ==================== ByteTrackNode 统计测试 ====================

class ByteTrackNodeStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_unique<ByteTrackNode>();
    }

    void TearDown() override {}

    std::unique_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeStatsTest, InitialStats) {
    auto stats = node_->stats();

    EXPECT_EQ(stats.processed_count, 0u);
    EXPECT_EQ(stats.error_count, 0u);
}

TEST_F(ByteTrackNodeStatsTest, StatsAfterProcessing) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    for (int i = 0; i < 5; ++i) {
        Frame frame = make_tracking_frame(i, dets);
        node_->process(frame);
    }

    auto stats = node_->stats();
    EXPECT_EQ(stats.processed_count, 5u);
}

// ==================== ByteTrackNode 边界测试 ====================

class ByteTrackNodeEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_unique<ByteTrackNode>();
    }

    void TearDown() override {}

    std::unique_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeEdgeCaseTest, ZeroAreaDetection) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.5f, 0.5f, 0.5f, 0.5f));

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, NegativeCoordinates) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(-0.1f, -0.1f, 0.3f, 0.3f));

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, CoordinatesGreaterThanOne) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.8f, 0.8f, 1.5f, 1.5f));

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, LargeFrameId) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

    Frame frame = make_tracking_frame(INT64_MAX, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, ManyDetections) {
    std::vector<Detection> dets;
    for (int i = 0; i < 100; ++i) {
        float x = (i % 10) * 0.1f;
        float y = (i / 10) * 0.1f;
        dets.push_back(make_detection_for_track(x, y, x + 0.08f, y + 0.08f, i % 5));
    }

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, VeryLowConfidence) {
    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.01f));

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node_->process(frame));
}

TEST_F(ByteTrackNodeEdgeCaseTest, HighConfidenceThreshold) {
    ByteTrackConfig config;
    config.track_thresh = 0.99f;  // 非常高的阈值
    auto node = std::make_unique<ByteTrackNode>(config);

    std::vector<Detection> dets;
    dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.95f));

    Frame frame = make_tracking_frame(0, dets);

    EXPECT_NO_THROW(node->process(frame));
}

// ==================== ByteTrackNode 并发测试 ====================

class ByteTrackNodeConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_shared<ByteTrackNode>();
    }

    void TearDown() override {}

    std::shared_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeConcurrencyTest, ConcurrentParamUpdate) {
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([this, &success_count, t]() {
            for (int i = 0; i < 100; ++i) {
                if (node_->set_param("track_thresh", static_cast<float>(t * 0.01 + i * 0.001))) {
                    ++success_count;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(success_count.load(), 0);
}

TEST_F(ByteTrackNodeConcurrencyTest, DISABLED_ConcurrentProcessAndParamUpdate) {
    std::atomic<bool> running{true};

    std::thread process_thread([this, &running]() {
        std::vector<Detection> dets;
        dets.push_back(make_detection_for_track(0.1f, 0.1f, 0.3f, 0.3f));

        int frame_id = 0;
        while (running.load()) {
            Frame frame = make_tracking_frame(frame_id++, dets);
            node_->process(frame);
            std::this_thread::sleep_for(1ms);
        }
    });

    std::thread param_thread([this, &running]() {
        while (running.load()) {
            node_->set_param("track_thresh", 0.3f + (rand() % 100) * 0.001f);
            std::this_thread::sleep_for(2ms);
        }
    });

    std::this_thread::sleep_for(100ms);
    running.store(false);

    process_thread.join();
    param_thread.join();

    // 应不崩溃
    EXPECT_TRUE(true);
}

// ==================== ByteTrackNode 队列测试 ====================

class ByteTrackNodeQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        node_ = std::make_unique<ByteTrackNode>();
    }

    void TearDown() override {}

    std::unique_ptr<ByteTrackNode> node_;
};

TEST_F(ByteTrackNodeQueueTest, CreateOutputQueue) {
    node_->create_output_queue(16, OverflowPolicy::DROP_OLDEST);

    auto output_queue = node_->output_queue();
    ASSERT_TRUE(output_queue != nullptr);
    EXPECT_EQ(output_queue->capacity(), 16u);
}

TEST_F(ByteTrackNodeQueueTest, IsSourceReturnsFalse) {
    EXPECT_FALSE(node_->is_source());
}

TEST_F(ByteTrackNodeQueueTest, IsSinkReturnsFalse) {
    EXPECT_FALSE(node_->is_sink());
}

}  // namespace
}  // namespace visionpipe
