#include "nodes/sink/json_result_sink.h"

#include <nlohmann/json.hpp>

#include "core/logger.h"

namespace visionpipe {

JsonResultSink::JsonResultSink(const JsonResultSinkConfig& config,
                               const std::string& name)
    : SinkNode(name)
    , config_(config)
    , json_queue_(std::make_shared<BoundedQueue<std::string>>(
          config.buffer_capacity, OverflowPolicy::DROP_OLDEST)) {}

void JsonResultSink::process(Frame& frame) {
    if (!enabled()) return;

    nlohmann::json j;
    j["stream_id"] = frame.stream_id;
    j["frame_id"] = frame.frame_id;
    j["pts_us"] = frame.pts_us;

    if (config_.include_detections) {
        nlohmann::json dets = nlohmann::json::array();
        for (const auto& d : frame.detections) {
            dets.push_back({
                {"bbox", {d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]}},
                {"class_id", d.class_id},
                {"confidence", d.confidence},
                {"track_id", d.track_id},
            });
        }
        j["detections"] = std::move(dets);
    }

    if (config_.include_tracks) {
        nlohmann::json tracks = nlohmann::json::array();
        for (const auto& t : frame.tracks) {
            tracks.push_back({
                {"track_id", t.track_id},
                {"class_id", t.class_id},
                {"bbox", {t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]}},
                {"age", t.age},
                {"confidence", t.confidence},
            });
        }
        j["tracks"] = std::move(tracks);
    }

    json_queue_->push(j.dump());
}

std::optional<std::string> JsonResultSink::pop_json(
    std::chrono::milliseconds timeout) {
    return json_queue_->pop_for(timeout);
}

}  // namespace visionpipe
