#include "bytetrack_node.h"

#include "core/logger.h"

namespace visionpipe {

ByteTrackNode::ByteTrackNode(const ByteTrackConfig& config,
                             const std::string& name)
    : NodeBase(name)
    , config_(config)
    , tracker_(std::make_unique<ByteTrackImpl>(
          config.track_thresh,
          config.track_buffer,
          config.match_thresh,
          config.frame_rate)) {
    create_output_queue(16, OverflowPolicy::DROP_OLDEST);
}

ByteTrackNode::~ByteTrackNode() = default;

void ByteTrackNode::process(Frame& frame) {
    if (!tracker_) {
        return;
    }

    // 将 Detection 转换为 TrackBox
    std::vector<TrackBox> boxes;
    boxes.reserve(frame.detections.size());

    for (const auto& det : frame.detections) {
        TrackBox box;
        box.bbox[0] = det.bbox[0];
        box.bbox[1] = det.bbox[1];
        box.bbox[2] = det.bbox[2];
        box.bbox[3] = det.bbox[3];
        box.class_id = det.class_id;
        box.confidence = det.confidence;
        box.track_id = det.track_id;
        boxes.push_back(box);
    }

    std::unordered_map<size_t, int64_t> det_track_map;
    auto active_tracks = tracker_->update(boxes, det_track_map);

    for (const auto& [det_idx, track_id] : det_track_map) {
        if (det_idx < frame.detections.size()) {
            frame.detections[det_idx].track_id = track_id;
        }
    }

    // 写入活跃轨迹到 frame.tracks
    frame.tracks.clear();
    frame.tracks.reserve(active_tracks.size());
    for (const auto& track : active_tracks) {
        Track t;
        t.track_id = track.id();
        t.class_id = track.class_id();
        t.bbox[0] = track.box().bbox[0];
        t.bbox[1] = track.box().bbox[1];
        t.bbox[2] = track.box().bbox[2];
        t.bbox[3] = track.box().bbox[3];
        t.age = track.age();
        t.confidence = track.confidence();
        frame.tracks.push_back(t);
    }

    ++processed_count_;
}

bool ByteTrackNode::set_param(const std::string& name, const ParamValue& value) {
    std::lock_guard<std::mutex> lock(params_mutex_);

    try {
        if (name == "track_thresh") {
            if (std::holds_alternative<float>(value)) {
                config_.track_thresh = std::get<float>(value);
                return true;
            } else if (std::holds_alternative<double>(value)) {
                config_.track_thresh = static_cast<float>(std::get<double>(value));
                return true;
            }
        } else if (name == "track_buffer") {
            if (std::holds_alternative<int>(value)) {
                config_.track_buffer = std::get<int>(value);
                return true;
            }
        } else if (name == "match_thresh") {
            if (std::holds_alternative<float>(value)) {
                config_.match_thresh = std::get<float>(value);
                return true;
            } else if (std::holds_alternative<double>(value)) {
                config_.match_thresh = static_cast<float>(std::get<double>(value));
                return true;
            }
        } else if (name == "frame_rate") {
            if (std::holds_alternative<int>(value)) {
                config_.frame_rate = std::get<int>(value);
                return true;
            }
        }
    } catch (const std::exception& e) {
        VP_LOG_ERROR("ByteTrackNode '{}': failed to set param '{}': {}",
                     name_, name, e.what());
    }

    return false;
}

void ByteTrackNode::reset() {
    if (tracker_) {
        tracker_->reset();
    }
}

size_t ByteTrackNode::active_track_count() const {
    return tracker_ ? tracker_->tracks().size() : 0;
}

}  // namespace visionpipe
