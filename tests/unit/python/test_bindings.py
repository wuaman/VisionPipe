from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe


def test_exports_are_available() -> None:
    assert visionpipe.Pipeline is not None
    assert visionpipe.PipelineBuilder is not None
    assert visionpipe.PipelineManager is not None
    assert visionpipe.Frame is not None
    assert visionpipe.FileSource is not None
    assert visionpipe.RtspSource is not None
    assert visionpipe.ByteTrackNode is not None
    assert visionpipe.IModelEngine is not None
    assert visionpipe.MockModelEngine is not None
    assert visionpipe.DetectorNode is not None
    assert visionpipe.ClassifierNode is not None
    assert visionpipe.YoloSegNode is not None
    assert visionpipe.JsonResultSink is not None
    assert visionpipe.MjpegSink is not None
    assert visionpipe.WebRTCSink is not None


def test_source_config_defaults_and_mutation() -> None:
    config = visionpipe.SourceConfig()
    assert config.decode_mode == visionpipe.DecodeMode.AUTO
    assert config.gpu_device == 0
    assert config.queue_capacity == 16
    assert config.overflow_policy == visionpipe.OverflowPolicy.DROP_OLDEST
    assert config.stream_id == 0

    config.uri = "video.mp4"
    config.decode_mode = visionpipe.DecodeMode.CPU
    config.gpu_device = 1
    config.queue_capacity = 32
    config.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    config.stream_id = 7

    assert config.uri == "video.mp4"
    assert config.decode_mode == visionpipe.DecodeMode.CPU
    assert config.gpu_device == 1
    assert config.queue_capacity == 32
    assert config.overflow_policy == visionpipe.OverflowPolicy.BLOCK
    assert config.stream_id == 7


def test_infer_configs_defaults_and_mutation() -> None:
    detector = visionpipe.DetectorConfig()
    assert detector.input_width == 640
    assert detector.input_height == 640
    assert detector.max_detections == 300
    assert detector.workers == 1

    detector.score_threshold = 0.4
    detector.nms_threshold = 0.3
    detector.max_detections = 12
    detector.workers = 2
    assert math.isclose(detector.score_threshold, 0.4, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(detector.nms_threshold, 0.3, rel_tol=0.0, abs_tol=1e-6)
    assert detector.max_detections == 12
    assert detector.workers == 2

    classifier = visionpipe.ClassifierConfig()
    assert classifier.input_width == 224
    assert classifier.input_height == 224
    assert classifier.max_batch_size == 32
    assert classifier.workers == 1
    assert classifier.normalize_mean_std is True

    classifier.max_batch_size = 8
    classifier.normalize_mean_std = False
    assert classifier.max_batch_size == 8
    assert classifier.normalize_mean_std is False

    segment = visionpipe.YoloSegConfig()
    assert segment.input_width == 640
    assert segment.input_height == 640
    assert segment.max_detections == 100
    assert segment.workers == 1

    segment.mask_threshold = 0.6
    segment.workers = 3
    assert math.isclose(segment.mask_threshold, 0.6, rel_tol=0.0, abs_tol=1e-6)
    assert segment.workers == 3


def test_detection_and_track_bbox_properties() -> None:
    detection = visionpipe.Detection()
    detection.bbox = [10.0, 20.0, 40.0, 60.0]
    detection.class_id = 3
    detection.confidence = 0.9
    detection.track_id = 12

    assert tuple(detection.bbox) == (10.0, 20.0, 40.0, 60.0)
    assert detection.width() == 30.0
    assert detection.height() == 40.0
    assert detection.area() == 1200.0
    assert detection.class_id == 3
    assert math.isclose(detection.confidence, 0.9, rel_tol=0.0, abs_tol=1e-6)
    assert detection.track_id == 12

    track = visionpipe.Track()
    track.bbox = (1.0, 2.0, 5.0, 8.0)
    track.track_id = 99
    track.class_id = 6
    track.age = 4
    track.confidence = 0.75

    assert tuple(track.bbox) == (1.0, 2.0, 5.0, 8.0)
    assert track.track_id == 99
    assert track.class_id == 6
    assert track.age == 4
    assert math.isclose(track.confidence, 0.75, rel_tol=0.0, abs_tol=1e-6)


def test_frame_clear_and_lists() -> None:
    frame = visionpipe.Frame()
    frame.stream_id = 5
    frame.frame_id = 8
    frame.pts_us = 123456

    detection = visionpipe.Detection()
    detection.bbox = [0.0, 0.0, 1.0, 1.0]
    track = visionpipe.Track()
    track.bbox = [0.0, 0.0, 2.0, 2.0]

    frame.detections = [detection]
    frame.tracks = [track]

    assert frame.has_image() is False
    assert len(frame.detections) == 1
    assert len(frame.tracks) == 1

    frame.clear()

    assert frame.stream_id == 0
    assert frame.frame_id == 0
    assert frame.pts_us == 0
    assert frame.detections == []
    assert frame.tracks == []
    assert frame.has_image() is False


def test_mock_engine_and_infer_node_construction() -> None:
    engine = visionpipe.MockModelEngine()

    assert isinstance(engine, visionpipe.IModelEngine)
    assert engine.device_memory_bytes() == 0
    assert engine.output_count() == 1

    detector = visionpipe.DetectorNode(engine)
    assert detector.name() == "detector"
    assert detector.worker_count() == 1
    assert detector.config().input_width == 640

    detector_config = visionpipe.DetectorConfig()
    detector_config.workers = 0
    named_detector = visionpipe.DetectorNode(engine, detector_config, "det-2")
    assert named_detector.name() == "det-2"
    assert named_detector.worker_count() == 1

    classifier_config = visionpipe.ClassifierConfig()
    classifier_config.workers = 0
    classifier = visionpipe.ClassifierNode(engine, classifier_config, "cls")
    assert classifier.name() == "cls"
    assert classifier.worker_count() == 1
    assert classifier.config().max_batch_size == 32

    segment_config = visionpipe.YoloSegConfig()
    segment_config.workers = 0
    segment = visionpipe.YoloSegNode(engine, segment_config, "seg")
    assert segment.name() == "seg"
    assert segment.worker_count() == 1
    assert segment.last_masks() == []


def test_infer_nodes_raise_for_missing_engine() -> None:
    try:
        visionpipe.DetectorNode(None)
    except TypeError:
        pass
    else:
        raise AssertionError("DetectorNode(None) should fail")

    try:
        visionpipe.ClassifierNode(None)
    except TypeError:
        pass
    else:
        raise AssertionError("ClassifierNode(None) should fail")

    try:
        visionpipe.YoloSegNode(None)
    except TypeError:
        pass
    else:
        raise AssertionError("YoloSegNode(None) should fail")


def test_detector_roi_binding() -> None:
    engine = visionpipe.MockModelEngine()
    detector = visionpipe.DetectorNode(engine)
    detector.set_roi([[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]])
    detector.clear_roi()


# NOTE: test_pipeline_builder_dsl_builds_graph removed — old behavior: >> returns
# PipelineBuilder with .build(). New spec: >> returns Pipeline directly.

# NOTE: test_pipeline_run_alias_calls_start removed — old behavior: run() is just
# start() alias. New spec: run(block=False) with new semantics.


def test_pipeline_graph_building_and_lookup() -> None:
    source = visionpipe.FileSource("video.mp4", visionpipe.DecodeMode.CPU)
    tracker = visionpipe.ByteTrackNode()

    config = visionpipe.PipelineConfig()
    config.name = "py-pipe"
    pipeline = visionpipe.Pipeline(config)

    returned = pipeline.add_node(source).add_node(tracker).connect(source, tracker)
    assert returned is pipeline

    pipeline.validate_dag()

    assert pipeline.name() == "py-pipe"
    assert pipeline.state() == visionpipe.PipelineState.INIT
    assert pipeline.get_node(source.name()) is source
    assert pipeline.get_node(tracker.name()) is tracker

    nodes = pipeline.nodes()
    assert set(nodes) == {source.name(), tracker.name()}
    assert nodes[source.name()] is source
    assert nodes[tracker.name()] is tracker

    source_nodes = pipeline.source_nodes()
    assert len(source_nodes) == 1
    assert source_nodes[0] is source

    stats = pipeline.stats()
    assert stats.state == visionpipe.PipelineState.INIT
    assert stats.total_frames_processed == 0
    assert stats.total_errors == 0
    assert pipeline.processed_count() == 0


def test_pipeline_manager_create_list_get() -> None:
    manager = visionpipe.PipelineManager()

    config = visionpipe.PipelineConfig()
    config.id = "pipe-config"
    config.name = "configured"

    created_id = manager.create(config)
    assert created_id == "pipe-config"
    assert manager.list() == ["pipe-config"]
    assert manager.status("pipe-config") == visionpipe.PipelineStatus.INIT
    assert manager.get("pipe-config").name() == "configured"

    pipeline = visionpipe.Pipeline()
    source = visionpipe.FileSource("video-2.mp4", visionpipe.DecodeMode.CPU)
    tracker = visionpipe.ByteTrackNode(name="tracker-2")
    pipeline.add_node(source).add_node(tracker).connect(source, tracker)

    second_id = manager.create_pipeline(pipeline)
    ids = manager.list()
    assert ids == sorted([created_id, second_id])
    assert manager.get(second_id) is pipeline
