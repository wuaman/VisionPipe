#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <chrono>
#include <optional>

#include "bindings.h"
#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "hal/imodel_engine.h"
#include "hal/nvidia/trt_model_engine.h"
#include "nodes/source/file_source.h"
#include "nodes/source/rtsp_source.h"
#include "nodes/tracker/bytetrack_node.h"
#include "nodes/infer/detector_node.h"
#include "nodes/infer/classifier_node.h"
#include "nodes/infer/segment_node.h"
#include "nodes/sink/json_result_sink.h"
#include "nodes/sink/mjpeg_sink.h"

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
        .def("create_output_queue", &NodeBase::create_output_queue,
             nb::arg("capacity") = 16,
             nb::arg("policy") = OverflowPolicy::DROP_OLDEST)
        .def("pop_frame", [](NodeBase& node, int timeout_ms) -> nb::object {
            auto q = node.output_queue();
            if (!q) return nb::none();
            auto result = q->pop_for(std::chrono::milliseconds(timeout_ms));
            if (!result.has_value()) return nb::none();
            return nb::cast(std::move(*result));
        }, nb::arg("timeout_ms") = 500);

    nb::class_<FileSource, NodeBase>(m, "FileSource")
        .def(nb::init<const SourceConfig&>(), nb::arg("config"))
        .def(nb::init<const std::string&, DecodeMode>(), nb::arg("uri"), nb::arg("mode") = DecodeMode::AUTO)
        .def("width", &FileSource::width)
        .def("height", &FileSource::height)
        .def("fps", &FileSource::fps)
        .def("frame_count", &FileSource::frame_count)
        .def("current_frame", &FileSource::current_frame)
        .def("actual_decode_mode", &FileSource::actual_decode_mode)
        .def("config", &FileSource::config, nb::rv_policy::reference_internal);

    nb::class_<RtspSource, NodeBase>(m, "RtspSource")
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
        .def("active_track_count", &ByteTrackNode::active_track_count);

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

    nb::class_<SegmentNode, NodeBase>(m, "SegmentNode")
        .def(nb::init<std::shared_ptr<IModelEngine>, const SegmentConfig&, const std::string&>(),
             nb::arg("engine"),
             nb::arg("config") = SegmentConfig(),
             nb::arg("name") = "segment")
        .def(nb::init<std::shared_ptr<IModelEngine>, const std::string&>(),
             nb::arg("engine"),
             nb::arg("name"))
        .def("config", &SegmentNode::config, nb::rv_policy::reference_internal)
        .def("worker_count", &SegmentNode::worker_count)
        .def("last_masks", &SegmentNode::last_masks, nb::rv_policy::reference_internal);

    nb::class_<JsonResultSinkConfig>(m, "JsonResultSinkConfig")
        .def(nb::init<>())
        .def_rw("buffer_capacity", &JsonResultSinkConfig::buffer_capacity)
        .def_rw("include_detections", &JsonResultSinkConfig::include_detections)
        .def_rw("include_tracks", &JsonResultSinkConfig::include_tracks);

    nb::class_<JsonResultSink, NodeBase>(m, "JsonResultSink")
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

    nb::class_<MjpegSink, NodeBase>(m, "MjpegSink")
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
}
