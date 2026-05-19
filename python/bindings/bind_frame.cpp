#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <stdexcept>

#include "bindings.h"
#include "core/tensor.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/pipeline.h"
#include "core/bounded_queue.h"
#include "nodes/source/source_config.h"
#include "nodes/tracker/bytetrack_node.h"
#include "nodes/infer/detector_node.h"
#include "nodes/infer/classifier_node.h"
#include "nodes/infer/segment_node.h"

namespace nb = nanobind;
using namespace visionpipe;

namespace {

nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>> frame_image_numpy(const Frame& frame) {
    if (!frame.has_image()) {
        throw std::runtime_error("Frame has no image");
    }
    const Tensor& t = frame.image;
    if (t.memory_type() != MemoryType::CPU) {
        throw std::runtime_error("image_numpy() only supports CPU tensors");
    }
    if (t.shape.size() != 3) {
        throw std::runtime_error("image_numpy() expects HWC tensor (ndim=3)");
    }
    size_t h = t.shape[0], w = t.shape[1], c = t.shape[2];
    return nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>(
        static_cast<uint8_t*>(t.data),
        {h, w, c},
        nb::handle()
    );
}

std::array<float, 4> get_detection_bbox(const Detection& detection) {
    return {detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]};
}

void set_detection_bbox(Detection& detection, const std::array<float, 4>& bbox) {
    for (size_t i = 0; i < bbox.size(); ++i) {
        detection.bbox[i] = bbox[i];
    }
}

std::array<float, 4> get_track_bbox(const Track& track) {
    return {track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]};
}

void set_track_bbox(Track& track, const std::array<float, 4>& bbox) {
    for (size_t i = 0; i < bbox.size(); ++i) {
        track.bbox[i] = bbox[i];
    }
}

}  // namespace

void bind_frame(nb::module_& m) {
    nb::class_<QueueStats>(m, "QueueStats")
        .def(nb::init<>())
        .def_rw("capacity", &QueueStats::capacity)
        .def_rw("current_size", &QueueStats::current_size)
        .def_rw("total_pushed", &QueueStats::total_pushed)
        .def_rw("total_popped", &QueueStats::total_popped)
        .def_rw("dropped_count", &QueueStats::dropped_count);

    nb::class_<NodeStats>(m, "NodeStats")
        .def(nb::init<>())
        .def_rw("processed_count", &NodeStats::processed_count)
        .def_rw("error_count", &NodeStats::error_count)
        .def_rw("fps", &NodeStats::fps)
        .def_rw("input_queue_stats", &NodeStats::input_queue_stats);

    nb::class_<PipelineConfig>(m, "PipelineConfig")
        .def(nb::init<>())
        .def_rw("name", &PipelineConfig::name)
        .def_rw("id", &PipelineConfig::id)
        .def_rw("default_queue_capacity", &PipelineConfig::default_queue_capacity)
        .def_rw("default_overflow_policy", &PipelineConfig::default_overflow_policy);

    nb::class_<PipelineStats>(m, "PipelineStats")
        .def(nb::init<>())
        .def_rw("state", &PipelineStats::state)
        .def_rw("total_frames_processed", &PipelineStats::total_frames_processed)
        .def_rw("total_errors", &PipelineStats::total_errors)
        .def_rw("node_stats", &PipelineStats::node_stats);

    nb::class_<SourceConfig>(m, "SourceConfig")
        .def(nb::init<>())
        .def(nb::init<const std::string&>())
        .def(nb::init<const std::string&, DecodeMode, int, size_t, OverflowPolicy, int64_t>(),
             nb::arg("uri"),
             nb::arg("decode_mode"),
             nb::arg("gpu_device") = 0,
             nb::arg("queue_capacity") = 16,
             nb::arg("overflow_policy") = OverflowPolicy::DROP_OLDEST,
             nb::arg("stream_id") = 0)
        .def_rw("uri", &SourceConfig::uri)
        .def_rw("decode_mode", &SourceConfig::decode_mode)
        .def_rw("gpu_device", &SourceConfig::gpu_device)
        .def_rw("queue_capacity", &SourceConfig::queue_capacity)
        .def_rw("overflow_policy", &SourceConfig::overflow_policy)
        .def_rw("stream_id", &SourceConfig::stream_id)
        .def_rw("loop", &SourceConfig::loop)
        .def_rw("skip_frames", &SourceConfig::skip_frames)
        .def_rw("max_retries", &SourceConfig::max_retries)
        .def_rw("retry_interval_ms", &SourceConfig::retry_interval_ms);

    nb::class_<ByteTrackConfig>(m, "ByteTrackConfig")
        .def(nb::init<>())
        .def_rw("track_thresh", &ByteTrackConfig::track_thresh)
        .def_rw("track_buffer", &ByteTrackConfig::track_buffer)
        .def_rw("match_thresh", &ByteTrackConfig::match_thresh)
        .def_rw("frame_rate", &ByteTrackConfig::frame_rate);

    nb::class_<DetectorConfig>(m, "DetectorConfig")
        .def(nb::init<>())
        .def_rw("input_width", &DetectorConfig::input_width)
        .def_rw("input_height", &DetectorConfig::input_height)
        .def_rw("score_threshold", &DetectorConfig::score_threshold)
        .def_rw("nms_threshold", &DetectorConfig::nms_threshold)
        .def_rw("max_detections", &DetectorConfig::max_detections)
        .def_rw("workers", &DetectorConfig::workers);

    nb::class_<ClassifierConfig>(m, "ClassifierConfig")
        .def(nb::init<>())
        .def_rw("input_width", &ClassifierConfig::input_width)
        .def_rw("input_height", &ClassifierConfig::input_height)
        .def_rw("max_batch_size", &ClassifierConfig::max_batch_size)
        .def_rw("workers", &ClassifierConfig::workers)
        .def_rw("normalize_mean_std", &ClassifierConfig::normalize_mean_std);

    nb::class_<SegmentConfig>(m, "SegmentConfig")
        .def(nb::init<>())
        .def_rw("input_width", &SegmentConfig::input_width)
        .def_rw("input_height", &SegmentConfig::input_height)
        .def_rw("score_threshold", &SegmentConfig::score_threshold)
        .def_rw("nms_threshold", &SegmentConfig::nms_threshold)
        .def_rw("mask_threshold", &SegmentConfig::mask_threshold)
        .def_rw("max_detections", &SegmentConfig::max_detections)
        .def_rw("workers", &SegmentConfig::workers);

    nb::class_<Detection>(m, "Detection")
        .def(nb::init<>())
        .def_prop_rw("bbox", &get_detection_bbox, &set_detection_bbox)
        .def_rw("class_id", &Detection::class_id)
        .def_rw("confidence", &Detection::confidence)
        .def_rw("track_id", &Detection::track_id)
        .def("width", &Detection::width)
        .def("height", &Detection::height)
        .def("area", &Detection::area);

    nb::class_<Track>(m, "Track")
        .def(nb::init<>())
        .def_prop_rw("bbox", &get_track_bbox, &set_track_bbox)
        .def_rw("track_id", &Track::track_id)
        .def_rw("class_id", &Track::class_id)
        .def_rw("age", &Track::age)
        .def_rw("confidence", &Track::confidence);

    nb::class_<Classification>(m, "Classification")
        .def(nb::init<>())
        .def_rw("detection_index", &Classification::detection_index)
        .def_rw("class_id", &Classification::class_id)
        .def_rw("confidence", &Classification::confidence);

    nb::class_<Frame>(m, "Frame", nb::dynamic_attr())
        .def(nb::init<>())
        .def_rw("stream_id", &Frame::stream_id)
        .def_rw("frame_id", &Frame::frame_id)
        .def_rw("pts_us", &Frame::pts_us)
        .def_rw("detections", &Frame::detections)
        .def_rw("classifications", &Frame::classifications)
        .def_rw("tracks", &Frame::tracks)
        .def("clear", &Frame::clear)
        .def("has_image", &Frame::has_image)
        .def("image_numpy", &frame_image_numpy);
}
