#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>

#include <chrono>
#include <optional>

#include "bindings.h"
#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/source_node.h"
#include "nodes/sink/sink_node.h"
#include "hal/imodel_engine.h"
#include "hal/nvidia/trt_model_engine.h"
#include "nodes/source/file_source.h"
#include "nodes/source/rtsp_source.h"
#include "nodes/tracker/bytetrack_node.h"
#include "nodes/infer/detector_node.h"
#include "nodes/infer/classifier_node.h"
#include "nodes/infer/yolo_seg_node.h"
#include "nodes/infer/rtmpose_node.h"
#include "nodes/infer/yolo_pose_node.h"
#include "nodes/sink/json_result_sink.h"
#include "nodes/sink/mjpeg_sink.h"
#include "nodes/sink/webrtc_sink.h"
#include "nodes/visualize/annotator_node.h"

namespace nb = nanobind;
using namespace visionpipe;

void bind_nodes(nb::module_& m) {
    nb::class_<IModelEngine>(m, "IModelEngine")
        .def("device_memory_bytes", &IModelEngine::device_memory_bytes)
        .def("output_count", &IModelEngine::output_count);

    nb::class_<MockModelEngine, IModelEngine>(m, "MockModelEngine")
        .def(nb::init<>());

    nb::class_<TrtModelEngine, IModelEngine>(m, "TrtModelEngine")
        .def(nb::init<const std::string&>(), nb::arg("engine_path"))
        .def("device_memory_bytes", &TrtModelEngine::device_memory_bytes)
        .def("output_count", &TrtModelEngine::output_count);

    nb::class_<NodeBase>(m, "NodeBase")
        .def("name", &NodeBase::name, nb::rv_policy::reference_internal)
        .def("state", &NodeBase::state)
        .def("start", &NodeBase::start)
        .def("stop", &NodeBase::stop, nb::arg("drain") = true)
        .def("wait_stop", &NodeBase::wait_stop)
        .def("stats", &NodeBase::stats)
        .def("is_source", &NodeBase::is_source)
        .def("is_sink", &NodeBase::is_sink)
        .def("set_param", &NodeBase::set_param,
             nb::arg("name"), nb::arg("value"))
        .def("create_output_queue", &NodeBase::create_output_queue,
             nb::arg("capacity") = 16,
             nb::arg("policy") = OverflowPolicy::DROP_OLDEST)
        .def("pop_frame", [](NodeBase& node, int timeout_ms) -> nb::object {
            auto q = node.output_queue();
            if (!q) return nb::none();
            auto result = q->pop_for(std::chrono::milliseconds(timeout_ms));
            if (!result.has_value()) return nb::none();
            return nb::cast(std::move(*result));
        }, nb::arg("timeout_ms") = 500)
        .def("input_queue_id", [](NodeBase& n) -> nb::object {
            auto* q = n.input_queue();
            if (!q) return nb::none();
            return nb::cast(reinterpret_cast<uintptr_t>(q));
        })
        .def("output_queue_id", [](NodeBase& n) -> nb::object {
            auto q = n.output_queue();
            if (!q) return nb::none();
            return nb::cast(reinterpret_cast<uintptr_t>(q.get()));
        });

    nb::class_<SourceNode, NodeBase>(m, "SourceNode");

    nb::class_<SinkNode, NodeBase>(m, "SinkNode")
        .def("enabled", &SinkNode::enabled)
        .def("set_enabled", &SinkNode::set_enabled, nb::arg("v"));

    nb::class_<FileSource, SourceNode>(m, "FileSource")
        .def(nb::init<const SourceConfig&>(), nb::arg("config"))
        .def(nb::init<const std::string&, DecodeMode>(), nb::arg("uri"), nb::arg("mode") = DecodeMode::AUTO)
        .def("width", &FileSource::width)
        .def("height", &FileSource::height)
        .def("fps", &FileSource::fps)
        .def("frame_count", &FileSource::frame_count)
        .def("current_frame", &FileSource::current_frame)
        .def("actual_decode_mode", &FileSource::actual_decode_mode)
        .def("config", &FileSource::config, nb::rv_policy::reference_internal);

    nb::class_<RtspSource, SourceNode>(m, "RtspSource")
        .def(nb::init<const SourceConfig&>(), nb::arg("config"))
        .def(nb::init<const std::string&, DecodeMode>(), nb::arg("uri"), nb::arg("mode") = DecodeMode::AUTO)
        .def("width", &RtspSource::width)
        .def("height", &RtspSource::height)
        .def("fps", &RtspSource::fps)
        .def("current_frame", &RtspSource::current_frame)
        .def("actual_decode_mode", &RtspSource::actual_decode_mode)
        .def("config", &RtspSource::config, nb::rv_policy::reference_internal)
        .def("is_connected", &RtspSource::is_connected);

    nb::class_<ByteTrackNode, NodeBase>(m, "ByteTrackNode")
        .def(nb::init<const ByteTrackConfig&, const std::string&>(),
             nb::arg("config") = ByteTrackConfig(),
             nb::arg("name") = "bytetrack")
        .def("config", &ByteTrackNode::config, nb::rv_policy::reference_internal)
        .def("reset", &ByteTrackNode::reset)
        .def("active_track_count", &ByteTrackNode::active_track_count)
        .def("set_param", &ByteTrackNode::set_param);

    nb::class_<DetectorNode, NodeBase>(m, "DetectorNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const DetectorConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = DetectorConfig(),
             nb::arg("name") = "detector")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &DetectorNode::config, nb::rv_policy::reference_internal)
        .def("set_roi", &DetectorNode::set_roi, nb::arg("polygons"))
        .def("clear_roi", &DetectorNode::clear_roi)
        .def("worker_count", &DetectorNode::worker_count);

    nb::class_<ClassifierNode, NodeBase>(m, "ClassifierNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const ClassifierConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = ClassifierConfig(),
             nb::arg("name") = "classifier")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &ClassifierNode::config, nb::rv_policy::reference_internal)
        .def("worker_count", &ClassifierNode::worker_count);

    nb::class_<YoloSegNode, NodeBase>(m, "YoloSegNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const YoloSegConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = YoloSegConfig(),
             nb::arg("name") = "yolo_seg")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &YoloSegNode::config, nb::rv_policy::reference_internal)
        .def("worker_count", &YoloSegNode::worker_count)
        .def("last_masks", &YoloSegNode::last_masks, nb::rv_policy::reference_internal);

    nb::class_<RtmPoseNode, NodeBase>(m, "RtmPoseNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const RtmPoseConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = RtmPoseConfig(),
             nb::arg("name") = "rtmpose")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &RtmPoseNode::config, nb::rv_policy::reference_internal)
        .def("worker_count", &RtmPoseNode::worker_count);

    nb::class_<YoloPoseNode, NodeBase>(m, "YoloPoseNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const YoloPoseConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = YoloPoseConfig(),
             nb::arg("name") = "yolo_pose")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &YoloPoseNode::config, nb::rv_policy::reference_internal)
        .def("worker_count", &YoloPoseNode::worker_count);

    nb::class_<JsonResultSinkConfig>(m, "JsonResultSinkConfig")
        .def(nb::init<>())
        .def_rw("buffer_capacity", &JsonResultSinkConfig::buffer_capacity)
        .def_rw("include_detections", &JsonResultSinkConfig::include_detections)
        .def_rw("include_tracks", &JsonResultSinkConfig::include_tracks)
        .def_rw("include_keypoints", &JsonResultSinkConfig::include_keypoints);

    nb::class_<JsonResultSink, SinkNode>(m, "JsonResultSink")
        .def(nb::init<const JsonResultSinkConfig&, const std::string&>(),
             nb::arg("config") = JsonResultSinkConfig(),
             nb::arg("name") = "json_result_sink")
        .def("config", &JsonResultSink::config, nb::rv_policy::reference_internal)
        .def("pop_json", [](JsonResultSink& sink, int timeout_ms) -> nb::object {
            auto result = sink.pop_json(std::chrono::milliseconds(timeout_ms));
            if (!result.has_value()) return nb::none();
            return nb::cast(std::move(*result));
        }, nb::arg("timeout_ms") = 500);

    nb::class_<MjpegSinkConfig>(m, "MjpegSinkConfig")
        .def(nb::init<>())
        .def_rw("jpeg_quality", &MjpegSinkConfig::jpeg_quality)
        .def_rw("buffer_capacity", &MjpegSinkConfig::buffer_capacity);

    nb::class_<MjpegSink, SinkNode>(m, "MjpegSink")
        .def(nb::init<const MjpegSinkConfig&, const std::string&>(),
             nb::arg("config") = MjpegSinkConfig(),
             nb::arg("name") = "mjpeg_sink")
        .def("config", &MjpegSink::config, nb::rv_policy::reference_internal)
        .def("pop_jpeg", [](MjpegSink& sink, int timeout_ms) -> nb::object {
            auto result = sink.pop_jpeg(std::chrono::milliseconds(timeout_ms));
            if (!result.has_value()) return nb::none();
            return nb::cast(nb::bytes(
                reinterpret_cast<const char*>(result->data()), result->size()));
        }, nb::arg("timeout_ms") = 500);

    nb::class_<AnnotatorConfig>(m, "AnnotatorConfig")
        .def(nb::init<>())
        .def_rw("draw_detections", &AnnotatorConfig::draw_detections)
        .def_rw("draw_tracks",     &AnnotatorConfig::draw_tracks)
        .def_rw("draw_masks",      &AnnotatorConfig::draw_masks)
        .def_rw("draw_keypoints",  &AnnotatorConfig::draw_keypoints)
        .def_rw("mask_alpha",      &AnnotatorConfig::mask_alpha)
        .def_rw("kpt_score_threshold", &AnnotatorConfig::kpt_score_threshold)
        .def_rw("class_names",     &AnnotatorConfig::class_names);

    nb::class_<AnnotatorNode, NodeBase>(m, "AnnotatorNode")
        .def(nb::init<const AnnotatorConfig&, const std::string&>(),
             nb::arg("config") = AnnotatorConfig(),
             nb::arg("name") = "annotator")
        .def("config", &AnnotatorNode::config, nb::rv_policy::reference_internal);

    nb::class_<WebRTCSinkConfig>(m, "WebRTCSinkConfig")
        .def(nb::init<>())
        .def_rw("video_bitrate_kbps", &WebRTCSinkConfig::video_bitrate_kbps)
        .def_rw("fps", &WebRTCSinkConfig::fps)
        .def_rw("keyframe_interval", &WebRTCSinkConfig::keyframe_interval)
        .def_rw("stun_server", &WebRTCSinkConfig::stun_server)
        .def_rw("use_nvenc", &WebRTCSinkConfig::use_nvenc);

    nb::class_<WebRTCSink, SinkNode>(m, "WebRTCSink")
        .def(nb::init<const WebRTCSinkConfig&, const std::string&>(),
             nb::arg("config") = WebRTCSinkConfig(),
             nb::arg("name") = "webrtc_sink")
        .def("config", &WebRTCSink::config, nb::rv_policy::reference_internal)
        .def("create_peer", &WebRTCSink::create_peer)
        .def("get_offer", [](WebRTCSink& sink, const std::string& peer_id, int timeout_ms) {
            return sink.get_offer(peer_id, std::chrono::milliseconds(timeout_ms));
        }, nb::arg("peer_id"), nb::arg("timeout_ms") = 10'000)
        .def("set_answer", &WebRTCSink::set_answer,
             nb::arg("peer_id"), nb::arg("sdp"))
        .def("add_candidate", &WebRTCSink::add_candidate,
             nb::arg("peer_id"), nb::arg("candidate"), nb::arg("mid"))
        .def("drain_candidates", &WebRTCSink::drain_candidates, nb::arg("peer_id"))
        .def("remove_peer", &WebRTCSink::remove_peer, nb::arg("peer_id"))
        .def("peer_count", &WebRTCSink::peer_count);
}
