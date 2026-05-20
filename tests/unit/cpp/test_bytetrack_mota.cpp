// test_bytetrack_mota.cpp
// T2.5 合成 MOTA 验证：50 帧 5 目标，可控噪声，MOTA > 0.6

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/frame.h"
#include "nodes/tracker/bytetrack_impl.h"
#include "nodes/tracker/bytetrack_node.h"

namespace visionpipe {
namespace {

struct GroundTruth {
    int64_t gt_id;
    float x1, y1, x2, y2;
    int class_id;
};

struct ObjectPath {
    int64_t gt_id;
    float start_cx, start_cy;
    float vx, vy;
    float width, height;
    int class_id;
};

std::vector<GroundTruth> generate_gt_frame(
    const std::vector<ObjectPath>& paths, int frame_idx) {
    std::vector<GroundTruth> gts;
    for (const auto& p : paths) {
        float cx = p.start_cx + p.vx * frame_idx;
        float cy = p.start_cy + p.vy * frame_idx;
        float half_w = p.width / 2.0f;
        float half_h = p.height / 2.0f;
        if (cx - half_w < 0.0f || cx + half_w > 1.0f ||
            cy - half_h < 0.0f || cy + half_h > 1.0f)
            continue;
        gts.push_back({p.gt_id, cx - half_w, cy - half_h,
                        cx + half_w, cy + half_h, p.class_id});
    }
    return gts;
}

std::vector<Detection> gt_to_detections(
    const std::vector<GroundTruth>& gts,
    std::mt19937& rng, float jitter_pct, float fn_prob) {
    std::uniform_real_distribution<float> jitter(-jitter_pct, jitter_pct);
    std::uniform_real_distribution<float> drop(0.0f, 1.0f);

    std::vector<Detection> dets;
    for (const auto& gt : gts) {
        if (drop(rng) < fn_prob) continue;

        float w = gt.x2 - gt.x1;
        float h = gt.y2 - gt.y1;
        Detection d;
        d.bbox[0] = gt.x1 + jitter(rng) * w;
        d.bbox[1] = gt.y1 + jitter(rng) * h;
        d.bbox[2] = gt.x2 + jitter(rng) * w;
        d.bbox[3] = gt.y2 + jitter(rng) * h;
        d.class_id = gt.class_id;
        d.confidence = 0.85f;
        d.track_id = -1;
        dets.push_back(d);
    }
    return dets;
}

struct MotaResult {
    double mota;
    int total_gt;
    int false_negatives;
    int false_positives;
    int id_switches;
};

MotaResult compute_mota(
    const std::vector<std::vector<GroundTruth>>& all_gts,
    const std::vector<std::vector<Track>>& all_tracks) {

    int total_gt = 0;
    int fn = 0;
    int fp = 0;
    int ids = 0;

    std::unordered_map<int64_t, int64_t> gt_to_track;

    for (size_t f = 0; f < all_gts.size(); ++f) {
        const auto& gts = all_gts[f];
        const auto& tracks = all_tracks[f];
        total_gt += static_cast<int>(gts.size());

        std::vector<bool> gt_matched(gts.size(), false);
        std::vector<bool> trk_matched(tracks.size(), false);

        for (size_t g = 0; g < gts.size(); ++g) {
            float best_iou = 0.3f;
            int best_t = -1;
            for (size_t t = 0; t < tracks.size(); ++t) {
                if (trk_matched[t]) continue;
                if (tracks[t].class_id != gts[g].class_id) continue;

                float ix1 = std::max(gts[g].x1, tracks[t].bbox[0]);
                float iy1 = std::max(gts[g].y1, tracks[t].bbox[1]);
                float ix2 = std::min(gts[g].x2, tracks[t].bbox[2]);
                float iy2 = std::min(gts[g].y2, tracks[t].bbox[3]);
                float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
                float area_g = (gts[g].x2 - gts[g].x1) * (gts[g].y2 - gts[g].y1);
                float area_t = (tracks[t].bbox[2] - tracks[t].bbox[0]) *
                               (tracks[t].bbox[3] - tracks[t].bbox[1]);
                float iou = inter / (area_g + area_t - inter + 1e-6f);

                if (iou > best_iou) {
                    best_iou = iou;
                    best_t = static_cast<int>(t);
                }
            }

            if (best_t >= 0) {
                gt_matched[g] = true;
                trk_matched[best_t] = true;

                int64_t gt_id = gts[g].gt_id;
                int64_t trk_id = tracks[best_t].track_id;
                auto it = gt_to_track.find(gt_id);
                if (it != gt_to_track.end() && it->second != trk_id) {
                    ++ids;
                }
                gt_to_track[gt_id] = trk_id;
            }
        }

        for (size_t g = 0; g < gts.size(); ++g)
            if (!gt_matched[g]) ++fn;
        for (size_t t = 0; t < tracks.size(); ++t)
            if (!trk_matched[t]) ++fp;
    }

    double mota = (total_gt > 0)
        ? 1.0 - static_cast<double>(fn + fp + ids) / total_gt
        : 0.0;

    return {mota, total_gt, fn, fp, ids};
}

TEST(ByteTrackMOTA, SyntheticFiveObjects) {
    std::vector<ObjectPath> paths = {
        {0, 0.15f, 0.2f,  0.005f,  0.002f, 0.12f, 0.15f, 0},
        {1, 0.50f, 0.3f, -0.003f,  0.004f, 0.10f, 0.12f, 0},
        {2, 0.80f, 0.5f, -0.006f, -0.001f, 0.14f, 0.14f, 1},
        {3, 0.30f, 0.7f,  0.004f, -0.003f, 0.10f, 0.10f, 1},
        {4, 0.60f, 0.8f,  0.002f, -0.005f, 0.11f, 0.13f, 0},
    };

    constexpr int kNumFrames = 50;
    constexpr float kJitter = 0.05f;
    constexpr float kFnProb = 0.02f;

    std::mt19937 rng(42);

    ByteTrackConfig cfg;
    cfg.track_thresh = 0.3f;
    cfg.track_buffer = 30;
    cfg.match_thresh = 0.3f;
    ByteTrackNode tracker(cfg);

    std::vector<std::vector<GroundTruth>> all_gts;
    std::vector<std::vector<Track>> all_tracks;

    for (int i = 0; i < kNumFrames; ++i) {
        auto gts = generate_gt_frame(paths, i);
        auto dets = gt_to_detections(gts, rng, kJitter, kFnProb);

        Frame f;
        f.frame_id = i;
        f.detections = std::move(dets);
        tracker.process(f);

        all_gts.push_back(std::move(gts));
        all_tracks.push_back(std::move(f.tracks));
    }

    auto result = compute_mota(all_gts, all_tracks);

    EXPECT_GT(result.mota, 0.6)
        << "MOTA=" << result.mota
        << " GT=" << result.total_gt
        << " FN=" << result.false_negatives
        << " FP=" << result.false_positives
        << " IDS=" << result.id_switches;
}

TEST(ByteTrackMOTA, NoJitterHighMota) {
    std::vector<ObjectPath> paths = {
        {0, 0.2f, 0.2f, 0.005f, 0.003f, 0.12f, 0.12f, 0},
        {1, 0.6f, 0.6f, -0.004f, 0.002f, 0.12f, 0.12f, 0},
    };

    constexpr int kNumFrames = 30;
    std::mt19937 rng(123);

    ByteTrackConfig cfg;
    cfg.track_thresh = 0.3f;
    ByteTrackNode tracker(cfg);

    std::vector<std::vector<GroundTruth>> all_gts;
    std::vector<std::vector<Track>> all_tracks;

    for (int i = 0; i < kNumFrames; ++i) {
        auto gts = generate_gt_frame(paths, i);
        auto dets = gt_to_detections(gts, rng, 0.0f, 0.0f);

        Frame f;
        f.frame_id = i;
        f.detections = std::move(dets);
        tracker.process(f);

        all_gts.push_back(std::move(gts));
        all_tracks.push_back(std::move(f.tracks));
    }

    auto result = compute_mota(all_gts, all_tracks);

    EXPECT_GT(result.mota, 0.85)
        << "MOTA=" << result.mota
        << " (perfect input should yield high MOTA)";
}

TEST(ByteTrackMOTA, HighJitterStillReasonable) {
    std::vector<ObjectPath> paths = {
        {0, 0.3f, 0.3f, 0.003f, 0.002f, 0.15f, 0.15f, 0},
        {1, 0.7f, 0.5f, -0.002f, 0.001f, 0.15f, 0.15f, 0},
    };

    constexpr int kNumFrames = 40;
    std::mt19937 rng(999);

    ByteTrackConfig cfg;
    cfg.track_thresh = 0.3f;
    cfg.track_buffer = 30;
    ByteTrackNode tracker(cfg);

    std::vector<std::vector<GroundTruth>> all_gts;
    std::vector<std::vector<Track>> all_tracks;

    for (int i = 0; i < kNumFrames; ++i) {
        auto gts = generate_gt_frame(paths, i);
        auto dets = gt_to_detections(gts, rng, 0.10f, 0.05f);

        Frame f;
        f.frame_id = i;
        f.detections = std::move(dets);
        tracker.process(f);

        all_gts.push_back(std::move(gts));
        all_tracks.push_back(std::move(f.tracks));
    }

    auto result = compute_mota(all_gts, all_tracks);

    EXPECT_GT(result.mota, 0.4)
        << "MOTA=" << result.mota
        << " (high noise should still produce reasonable tracking)";
}

}  // namespace
}  // namespace visionpipe
