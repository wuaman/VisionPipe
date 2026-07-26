// test_segment_tracker.cpp
// T2.5 集成测试：YoloSegNode + ByteTrackNode 联合验证

#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/tensor.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/post/seg_mask_decoder.h"
#include "nodes/infer/yolo_seg_node.h"
#include "nodes/tracker/bytetrack_node.h"

namespace visionpipe {
namespace {

class CpuTestAllocator : public IAllocator {
public:
    void* alloc(size_t bytes) override {
        auto* p = new(std::nothrow) uint8_t[bytes];
        if (p) std::memset(p, 0, bytes);
        return p;
    }
    void free(void* ptr) override { delete[] static_cast<uint8_t*>(ptr); }
    MemoryType type() const override { return MemoryType::CPU; }
};

static CpuTestAllocator g_alloc;

class MockSegCtx final : public IExecContext {
public:
    MockSegCtx(float cx, float cy, float w, float h, int cls, float conf)
        : cx_(cx), cy_(cy), w_(w), h_(h), cls_(cls), conf_(conf) {}

    void infer(const Tensor&, Tensor&) override {
        throw std::runtime_error("use infer_multi");
    }

    void infer_multi(const Tensor&, std::vector<Tensor>& outputs) override {
        outputs.clear();
        outputs.resize(2);

        const int num_anchors = 100;
        const int num_masks = 32;
        const int det_ch = 84 + num_masks;
        outputs[0] = Tensor({1, det_ch, num_anchors}, DataType::FLOAT32, &g_alloc);
        auto* det = static_cast<float*>(outputs[0].data);
        std::fill(det, det + outputs[0].numel(), 0.0f);

        det[0 * num_anchors] = cx_;
        det[1 * num_anchors] = cy_;
        det[2 * num_anchors] = w_;
        det[3 * num_anchors] = h_;
        det[(4 + cls_) * num_anchors] = conf_;

        for (int m = 0; m < num_masks; ++m)
            det[(84 + m) * num_anchors] = 0.5f;

        const int mh = 160, mw = 160;
        outputs[1] = Tensor({1, num_masks, mh, mw}, DataType::FLOAT32, &g_alloc);
        auto* proto = static_cast<float*>(outputs[1].data);
        int x1 = static_cast<int>((cx_ - w_ / 2) * mw / 640.0f);
        int y1 = static_cast<int>((cy_ - h_ / 2) * mh / 640.0f);
        int x2 = static_cast<int>((cx_ + w_ / 2) * mw / 640.0f);
        int y2 = static_cast<int>((cy_ + h_ / 2) * mh / 640.0f);
        for (int m = 0; m < num_masks; ++m)
            for (int y = 0; y < mh; ++y)
                for (int x = 0; x < mw; ++x) {
                    bool inside = (x >= x1 && x < x2 && y >= y1 && y < y2);
                    proto[m * mh * mw + y * mw + x] = inside ? 1.0f : -1.0f;
                }
    }

private:
    float cx_, cy_, w_, h_;
    int cls_;
    float conf_;
};

class MockSegEngine final : public IModelEngine {
public:
    MockSegEngine(float cx, float cy, float w, float h, int cls, float conf)
        : cx_(cx), cy_(cy), w_(w), h_(h), cls_(cls), conf_(conf) {}

    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<MockSegCtx>(cx_, cy_, w_, h_, cls_, conf_);
    }
    size_t device_memory_bytes() const override { return 0; }
    size_t output_count() const override { return 2; }

private:
    float cx_, cy_, w_, h_;
    int cls_;
    float conf_;
};

Frame make_frame(int64_t id, int w = 640, int h = 640) {
    Frame f;
    f.frame_id = id;
    f.stream_id = 0;
    f.pts_us = id * 40000;
    f.image = Tensor({h, w, 3}, DataType::UINT8, &g_alloc);
    return f;
}

std::vector<Frame> run_segment_pipeline(
    std::shared_ptr<IModelEngine> engine, int num_frames) {
    auto seg = std::make_shared<YoloSegNode>(engine, YoloSegConfig(), "seg_it");
    auto input_q = std::make_unique<BoundedQueue<Frame>>(32, OverflowPolicy::BLOCK);
    seg->set_input_queue(input_q.get());
    seg->create_output_queue(32, OverflowPolicy::BLOCK);
    seg->start();

    for (int i = 0; i < num_frames; ++i)
        input_q->push(make_frame(i));

    seg->stop(true);
    input_q->stop();
    seg->wait_stop();

    std::vector<Frame> results;
    auto out_q = seg->output_queue();
    while (auto f = out_q->pop()) results.push_back(std::move(*f));
    return results;
}

// ==================== 集成测试 ====================

TEST(SegmentTrackerIT, BasicPipeline) {
    auto engine = std::make_shared<MockSegEngine>(320, 320, 120, 120, 0, 0.9f);
    auto frames = run_segment_pipeline(engine, 10);

    ByteTrackConfig bt;
    bt.track_thresh = 0.3f;
    ByteTrackNode tracker(bt);

    int frames_with_tracks = 0;
    for (auto& f : frames) {
        ASSERT_FALSE(f.detections.empty()) << "Frame " << f.frame_id;
        ASSERT_FALSE(f.masks.empty()) << "Frame " << f.frame_id;
        EXPECT_EQ(f.detections.size(), f.masks.size());

        tracker.process(f);
        if (!f.tracks.empty()) ++frames_with_tracks;
    }
    EXPECT_GE(frames_with_tracks, 8);
}

TEST(SegmentTrackerIT, MaskBboxIouAbove90) {
    const int mw = 10, mh = 10;
    std::vector<uint8_t> mask(mw * mh, 255);
    float bbox[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    float iou = SegMaskDecoder::compute_mask_bbox_iou(mask, bbox, mw, mh);
    EXPECT_GT(iou, 0.9f);

    std::vector<uint8_t> partial(mw * mh, 0);
    for (int y = 0; y < mh; ++y)
        for (int x = 0; x < mw; ++x)
            if (x >= 1 && x < 9 && y >= 1 && y < 9) partial[y * mw + x] = 255;
    float iou2 = SegMaskDecoder::compute_mask_bbox_iou(partial, bbox, mw, mh);
    EXPECT_GT(iou2, 0.5f);
}

TEST(SegmentTrackerIT, TrackIdConsistency) {
    auto engine = std::make_shared<MockSegEngine>(320, 320, 120, 120, 0, 0.9f);
    auto frames = run_segment_pipeline(engine, 10);

    ByteTrackConfig bt;
    bt.track_thresh = 0.3f;
    ByteTrackNode tracker(bt);

    int64_t first_id = -1;
    int consistent_count = 0;
    for (auto& f : frames) {
        tracker.process(f);
        if (f.tracks.empty()) continue;
        if (first_id < 0) { first_id = f.tracks[0].track_id; continue; }
        for (const auto& t : f.tracks) {
            if (t.track_id == first_id) { ++consistent_count; break; }
        }
    }
    EXPECT_GE(first_id, 0);
    EXPECT_GE(consistent_count, 7);
}

TEST(SegmentTrackerIT, OcclusionRecovery) {
    ByteTrackConfig cfg;
    cfg.track_thresh = 0.3f;
    cfg.track_buffer = 30;
    ByteTrackNode tracker(cfg);

    auto make_det = [](float x1, float y1, float x2, float y2, int cls, float conf) {
        Detection d;
        d.bbox[0] = x1; d.bbox[1] = y1; d.bbox[2] = x2; d.bbox[3] = y2;
        d.class_id = cls; d.confidence = conf;
        return d;
    };

    int64_t tid = -1;
    for (int i = 0; i < 5; ++i) {
        Frame f;
        f.frame_id = i;
        f.detections.push_back(make_det(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.8f));
        tracker.process(f);
        if (!f.tracks.empty() && tid < 0) tid = f.tracks[0].track_id;
    }
    ASSERT_GE(tid, 0);

    for (int i = 5; i < 7; ++i) {
        Frame f; f.frame_id = i; tracker.process(f);
    }

    Frame f;
    f.frame_id = 7;
    f.detections.push_back(make_det(0.1f, 0.1f, 0.3f, 0.3f, 0, 0.8f));
    tracker.process(f);

    bool found = false;
    for (const auto& t : f.tracks)
        if (t.track_id == tid) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST(SegmentTrackerIT, MultiClassSegregation) {
    ByteTrackConfig cfg;
    cfg.track_thresh = 0.3f;
    ByteTrackNode tracker(cfg);

    for (int i = 0; i < 5; ++i) {
        Frame f;
        f.frame_id = i;
        Detection d1, d2;
        d1.bbox[0] = 0.1f; d1.bbox[1] = 0.1f; d1.bbox[2] = 0.4f; d1.bbox[3] = 0.4f;
        d1.class_id = 0; d1.confidence = 0.9f;
        d2.bbox[0] = 0.15f; d2.bbox[1] = 0.15f; d2.bbox[2] = 0.45f; d2.bbox[3] = 0.45f;
        d2.class_id = 1; d2.confidence = 0.85f;
        f.detections = {d1, d2};
        tracker.process(f);

        if (i >= 1 && f.tracks.size() >= 2) {
            std::unordered_set<int64_t> ids;
            for (const auto& t : f.tracks) ids.insert(t.track_id);
            EXPECT_EQ(ids.size(), f.tracks.size());
        }
    }
}

}  // namespace
}  // namespace visionpipe
