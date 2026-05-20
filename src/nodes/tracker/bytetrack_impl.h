#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nodes/tracker/kalman_filter.h"

namespace visionpipe {

struct TrackBox {
    float bbox[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int class_id = 0;
    float confidence = 0.0f;
    int64_t track_id = -1;

    float cx() const { return (bbox[0] + bbox[2]) / 2.0f; }
    float cy() const { return (bbox[1] + bbox[3]) / 2.0f; }
    float width() const { return bbox[2] - bbox[0]; }
    float height() const { return bbox[3] - bbox[1]; }
    float area() const { return width() * height(); }

    float iou(const TrackBox& other) const {
        float x1 = std::max(bbox[0], other.bbox[0]);
        float y1 = std::max(bbox[1], other.bbox[1]);
        float x2 = std::min(bbox[2], other.bbox[2]);
        float y2 = std::min(bbox[3], other.bbox[3]);
        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float union_area = area() + other.area() - intersection;
        return union_area > 0.0f ? intersection / union_area : 0.0f;
    }
};

class TrackedObject {
public:
    enum class State { New, Tracked, Lost, Removed };

    TrackedObject(int64_t id, const TrackBox& box)
        : id_(id)
        , state_(State::New)
        , class_id_(box.class_id)
        , confidence_(box.confidence)
        , box_(box) {
        float cx, cy, s, r;
        KalmanBoxTracker::bbox_to_xysr(box.bbox, cx, cy, s, r);
        kalman_ = KalmanBoxTracker(cx, cy, s, r);
    }

    int64_t id() const { return id_; }
    State state() const { return state_; }
    const TrackBox& box() const { return box_; }
    int class_id() const { return class_id_; }
    float confidence() const { return confidence_; }
    int age() const { return age_; }
    int hit_streak() const { return hit_streak_; }
    int miss_count() const { return miss_count_; }

    void update(const TrackBox& box) {
        float cx, cy, s, r;
        KalmanBoxTracker::bbox_to_xysr(box.bbox, cx, cy, s, r);
        kalman_.update(cx, cy, s, r);
        kalman_.get_bbox(box_.bbox);
        box_.class_id = box.class_id;
        box_.confidence = box.confidence;

        state_ = State::Tracked;
        ++hit_streak_;
        miss_count_ = 0;
        confidence_ = box.confidence;
    }

    void mark_lost() {
        state_ = State::Lost;
        hit_streak_ = 0;
        ++miss_count_;
    }

    void mark_removed() { state_ = State::Removed; }

    void predict() {
        ++age_;
        kalman_.predict();
        kalman_.get_bbox(box_.bbox);
    }

    void increment_miss() { ++miss_count_; }

private:
    int64_t id_;
    TrackBox box_;
    KalmanBoxTracker kalman_{0, 0, 1, 1};
    State state_ = State::New;
    int class_id_ = 0;
    float confidence_ = 0.0f;
    int age_ = 1;
    int hit_streak_ = 0;
    int miss_count_ = 0;
};

class ByteTrackImpl {
public:
    ByteTrackImpl(float track_thresh = 0.5f,
                  int track_buffer = 30,
                  float match_thresh = 0.3f,
                  int frame_rate = 30)
        : track_thresh_(track_thresh)
        , track_buffer_(track_buffer)
        , match_thresh_(match_thresh)
        , frame_rate_(frame_rate)
        , max_time_lost_(static_cast<int>(frame_rate / 30.0f * track_buffer)) {
    }

    std::vector<TrackedObject> update(const std::vector<TrackBox>& detections) {
        std::unordered_map<size_t, int64_t> unused;
        return update(detections, unused);
    }

    std::vector<TrackedObject> update(const std::vector<TrackBox>& detections,
                                      std::unordered_map<size_t, int64_t>& det_track_map) {
        det_track_map.clear();
        ++frame_id_;

        std::vector<TrackBox> high_dets, low_dets;
        std::vector<size_t> high_orig_idx, low_orig_idx;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].confidence >= track_thresh_) {
                high_dets.push_back(detections[i]);
                high_orig_idx.push_back(i);
            } else {
                low_dets.push_back(detections[i]);
                low_orig_idx.push_back(i);
            }
        }

        for (auto& trk : tracks_) {
            trk.predict();
        }

        std::vector<TrackedObject*> active_tracks, lost_tracks;
        for (auto& trk : tracks_) {
            if (trk.state() == TrackedObject::State::Tracked ||
                trk.state() == TrackedObject::State::New) {
                active_tracks.push_back(&trk);
            } else if (trk.state() == TrackedObject::State::Lost) {
                lost_tracks.push_back(&trk);
            }
        }

        // Pass 1: high-conf detections ↔ active tracks
        auto high_matched = linear_assignment(active_tracks, high_dets, match_thresh_);

        std::vector<TrackedObject*> unmatched_active;
        for (auto* trk : active_tracks) {
            if (high_matched.track_indices.find(trk->id()) == high_matched.track_indices.end()) {
                unmatched_active.push_back(trk);
            }
        }

        std::vector<TrackBox> unmatched_high_dets;
        std::vector<size_t> unmatched_high_orig_idx;
        for (size_t i = 0; i < high_dets.size(); ++i) {
            if (high_matched.det_indices.find(i) == high_matched.det_indices.end()) {
                unmatched_high_dets.push_back(high_dets[i]);
                unmatched_high_orig_idx.push_back(high_orig_idx[i]);
            }
        }

        // Pass 2: low-conf detections ↔ (unmatched active + lost tracks)
        std::vector<TrackedObject*> pass2_tracks;
        pass2_tracks.insert(pass2_tracks.end(), unmatched_active.begin(), unmatched_active.end());
        pass2_tracks.insert(pass2_tracks.end(), lost_tracks.begin(), lost_tracks.end());

        auto low_matched = linear_assignment(pass2_tracks, low_dets, match_thresh_);

        // Apply high-conf matches
        for (const auto& [track_id, det_idx] : high_matched.matches) {
            for (auto& trk : tracks_) {
                if (trk.id() == track_id) {
                    trk.update(high_dets[det_idx]);
                    det_track_map[high_orig_idx[det_idx]] = track_id;
                    break;
                }
            }
        }

        // Apply low-conf matches
        for (const auto& [track_id, det_idx] : low_matched.matches) {
            for (auto& trk : tracks_) {
                if (trk.id() == track_id) {
                    trk.update(low_dets[det_idx]);
                    det_track_map[low_orig_idx[det_idx]] = track_id;
                    break;
                }
            }
        }

        // Unmatched active tracks after both passes → lost
        for (auto* trk : unmatched_active) {
            if (low_matched.track_indices.find(trk->id()) == low_matched.track_indices.end()) {
                trk->mark_lost();
            }
        }

        // Pass 3: unmatched high-conf dets ↔ lost tracks (occlusion recovery)
        std::vector<TrackedObject*> still_lost;
        for (auto* trk : lost_tracks) {
            if (low_matched.track_indices.find(trk->id()) == low_matched.track_indices.end()) {
                still_lost.push_back(trk);
            }
        }

        auto lost_matched = linear_assignment(still_lost, unmatched_high_dets, match_thresh_);

        for (const auto& [track_id, det_idx] : lost_matched.matches) {
            for (auto& trk : tracks_) {
                if (trk.id() == track_id) {
                    trk.update(unmatched_high_dets[det_idx]);
                    det_track_map[unmatched_high_orig_idx[det_idx]] = track_id;
                    break;
                }
            }
        }

        std::vector<TrackBox> final_unmatched_high;
        for (size_t i = 0; i < unmatched_high_dets.size(); ++i) {
            if (lost_matched.det_indices.find(i) == lost_matched.det_indices.end()) {
                final_unmatched_high.push_back(unmatched_high_dets[i]);
            }
        }

        // Still-lost tracks → increment miss or remove
        for (auto* trk : still_lost) {
            if (lost_matched.track_indices.find(trk->id()) == lost_matched.track_indices.end()) {
                trk->increment_miss();
                if (trk->miss_count() > max_time_lost_) {
                    trk->mark_removed();
                }
            }
        }

        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [](const TrackedObject& t) { return t.state() == TrackedObject::State::Removed; }),
            tracks_.end());

        for (const auto& det : final_unmatched_high) {
            tracks_.emplace_back(next_id_++, det);
        }

        std::vector<TrackedObject> active;
        for (const auto& trk : tracks_) {
            if (trk.state() == TrackedObject::State::Tracked) {
                active.push_back(trk);
            } else if (trk.state() == TrackedObject::State::New && trk.hit_streak() >= 1) {
                active.push_back(trk);
            }
        }

        return active;
    }

    const std::deque<TrackedObject>& tracks() const { return tracks_; }

    void reset() {
        tracks_.clear();
        frame_id_ = 0;
        next_id_ = 0;
    }

private:
    struct MatchResult {
        std::unordered_map<int64_t, size_t> matches;
        std::unordered_set<int64_t> track_indices;
        std::unordered_set<size_t> det_indices;
    };

    MatchResult linear_assignment(std::vector<TrackedObject*>& trks,
                                  const std::vector<TrackBox>& dets,
                                  float thresh) {
        MatchResult result;
        if (trks.empty() || dets.empty()) return result;

        std::vector<std::tuple<float, size_t, size_t>> ious;
        for (size_t t = 0; t < trks.size(); ++t) {
            for (size_t d = 0; d < dets.size(); ++d) {
                if (trks[t]->class_id() != dets[d].class_id) continue;
                float iou_val = trks[t]->box().iou(dets[d]);
                if (iou_val > thresh) {
                    ious.emplace_back(iou_val, t, d);
                }
            }
        }

        std::sort(ious.begin(), ious.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

        std::unordered_set<size_t> used_tracks, used_dets;
        for (const auto& [iou_val, t, d] : ious) {
            if (used_tracks.count(t) || used_dets.count(d)) continue;
            int64_t tid = trks[t]->id();
            result.matches[tid] = d;
            result.track_indices.insert(tid);
            result.det_indices.insert(d);
            used_tracks.insert(t);
            used_dets.insert(d);
        }

        return result;
    }

    float track_thresh_;
    int track_buffer_;
    float match_thresh_;
    int frame_rate_;
    int max_time_lost_;

    std::deque<TrackedObject> tracks_;
    int frame_id_ = 0;
    int64_t next_id_ = 0;
};

}  // namespace visionpipe
