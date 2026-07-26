"""Real two-stage e2e test: YOLOv8 detection + ResNet50 second-stage classification.

This actually loads the FP16 TRT engines and runs inference end-to-end,
asserting that frame.classifications gets populated with real logits.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import visionpipe as vp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "tests" / "models"
DATA_DIR = PROJECT_ROOT / "tests" / "data"

YOLO_ENGINE = MODELS_DIR / "yolov8n_fp16.engine"
RESNET_ENGINE = MODELS_DIR / "resnet50_fp16.engine"
EFFNET_ENGINE = MODELS_DIR / "efficientnet_b0_fp16.engine"
SHUFFLE_ENGINE = MODELS_DIR / "shufflenetv2_fp16.engine"
SEG_ENGINE = MODELS_DIR / "yolov8m-seg_fp16.engine"
VIDEO = DATA_DIR / "48-3.mp4"


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing test asset: {p}")


def _make_source(uri: str, *, loop: bool = False) -> vp.FileSource:
    cfg = vp.SourceConfig(uri)
    cfg.decode_mode = vp.DecodeMode.AUTO
    cfg.loop = loop
    cfg.queue_capacity = 8
    cfg.overflow_policy = vp.OverflowPolicy.BLOCK
    return vp.FileSource(cfg)


def _safe_stop(manager, pid):
    try:
        if manager.status(pid) != vp.PipelineStatus.STOPPED:
            manager.stop(pid)
    except Exception:
        pass
    try:
        manager.destroy(pid)
    except Exception:
        pass


class _Collector(vp.PyNode):
    """Tail PyNode that snapshots classifications + masks per frame."""

    def __init__(self, name: str = "collector", max_frames: int = 200):
        super().__init__(name)
        self.captured: list = []
        self.lock = threading.Lock()
        self.max_frames = max_frames

    def process(self, frame):
        with self.lock:
            if len(self.captured) < self.max_frames:
                self.captured.append({
                    "frame_id": frame.frame_id,
                    "n_det": len(frame.detections),
                    "n_cls": len(frame.classifications),
                    "n_masks": len(frame.masks),
                    "cls": [
                        (c.detection_index, c.class_id, c.confidence)
                        for c in frame.classifications
                    ],
                    "det_classes": [d.class_id for d in frame.detections],
                    "det_scores": [d.confidence for d in frame.detections],
                    "mask_shapes": [m.shape if hasattr(m, "shape") else len(m) for m in frame.masks],
                })
        return frame


def _run_pipeline_briefly(pipeline, *, run_seconds: float = 8.0):
    manager = vp.PipelineManager()
    pid = manager.create_pipeline(pipeline)
    try:
        manager.start(pid)
        # 等待 source 自然结束 OR 超时
        deadline = time.monotonic() + run_seconds
        while time.monotonic() < deadline:
            if manager.status(pid) != vp.PipelineStatus.RUNNING:
                break
            time.sleep(0.1)
    finally:
        _safe_stop(manager, pid)


# --- 1) Engine load sanity ---

def test_classifier_engine_loads_resnet50():
    _require(RESNET_ENGINE)
    assert vp.TrtModelEngine(str(RESNET_ENGINE)) is not None


def test_classifier_engine_loads_efficientnet():
    _require(EFFNET_ENGINE)
    assert vp.TrtModelEngine(str(EFFNET_ENGINE)) is not None


def test_classifier_engine_loads_shufflenet():
    _require(SHUFFLE_ENGINE)
    assert vp.TrtModelEngine(str(SHUFFLE_ENGINE)) is not None


# --- 2) Mode 2: whole-image classify ---

def test_whole_image_classify_resnet50_produces_real_logits():
    _require(VIDEO, RESNET_ENGINE)

    src = _make_source(str(VIDEO))
    cls_cfg = vp.ClassifierConfig()
    cls_cfg.input_width = 224
    cls_cfg.input_height = 224
    cls_cfg.target_classes = []
    cls_engine = vp.TrtModelEngine(str(RESNET_ENGINE))
    classifier = vp.ClassifierNode(cls_engine, cls_cfg, "cls")
    collector = _Collector("tail")

    pipeline = vp.Pipeline(vp.PipelineConfig())
    pipeline.add_node(src)
    pipeline.add_node(classifier)
    pipeline.add_node(collector._cpp_node)
    pipeline.connect(src, classifier)
    pipeline.connect(classifier, collector._cpp_node)

    _run_pipeline_briefly(pipeline, run_seconds=15.0)

    captured = collector.captured
    assert len(captured) > 0, "no frames flowed through pipeline"
    with_cls = [c for c in captured if c["n_cls"] > 0]
    assert len(with_cls) > 0, (
        f"ResNet50 produced 0 classifications across {len(captured)} frames"
    )
    det_idx, cid, conf = with_cls[0]["cls"][0]
    assert det_idx == -1, f"whole-image mode should set detection_index=-1, got {det_idx}"
    assert 0 <= cid < 1000, f"class_id={cid} out of [0,1000) ImageNet range"
    assert 0.0 <= conf <= 1.0, f"confidence={conf} out of [0,1]"


# --- 3) Mode 1: detector → second-stage classify ---

def test_detector_to_resnet50_two_stage_real_inference():
    _require(VIDEO, YOLO_ENGINE, RESNET_ENGINE)

    src = _make_source(str(VIDEO))

    det_engine = vp.TrtModelEngine(str(YOLO_ENGINE))
    det_cfg = vp.DetectorConfig()
    det_cfg.score_threshold = 0.25
    det_cfg.nms_threshold = 0.45
    detector = vp.DetectorNode(det_engine, det_cfg, "det")

    cls_engine = vp.TrtModelEngine(str(RESNET_ENGINE))
    cls_cfg = vp.ClassifierConfig()
    cls_cfg.input_width = 224
    cls_cfg.input_height = 224
    cls_cfg.target_classes = [0]  # person only
    classifier = vp.ClassifierNode(cls_engine, cls_cfg, "cls")

    collector = _Collector("tail")

    pipeline = vp.Pipeline(vp.PipelineConfig())
    pipeline.add_node(src)
    pipeline.add_node(detector)
    pipeline.add_node(classifier)
    pipeline.add_node(collector._cpp_node)
    pipeline.connect(src, detector)
    pipeline.connect(detector, classifier)
    pipeline.connect(classifier, collector._cpp_node)

    _run_pipeline_briefly(pipeline, run_seconds=20.0)

    captured = collector.captured
    assert len(captured) > 0, "no frames flowed through pipeline"

    frames_with_person = [c for c in captured if 0 in c["det_classes"]]
    if not frames_with_person:
        pytest.skip(
            f"no person detections across {len(captured)} frames — "
            "cannot exercise two-stage mode"
        )

    frames_with_cls = [c for c in frames_with_person if c["n_cls"] > 0]
    assert len(frames_with_cls) > 0, (
        f"Two-stage produced 0 classifications across "
        f"{len(frames_with_person)} frames-with-person"
    )

    sample = frames_with_cls[0]
    n_persons = sum(1 for cid in sample["det_classes"] if cid == 0)
    assert sample["n_cls"] == n_persons, (
        f"expected 1 classification per person detection, "
        f"got n_cls={sample['n_cls']} n_persons={n_persons}"
    )
    for det_idx, cid, conf in sample["cls"]:
        assert 0 <= det_idx < sample["n_det"], (
            f"detection_index={det_idx} out of bounds [0,{sample['n_det']})"
        )
        assert sample["det_classes"][det_idx] == 0, (
            f"classified detection should be class 0 (person), got {sample['det_classes'][det_idx]}"
        )
        assert 0 <= cid < 1000
        assert 0.0 <= conf <= 1.0


# --- 4) EfficientNet / ShuffleNet real inference (whole-image) ---

def _run_whole_image_classify(engine_path: Path, run_seconds: float = 15.0) -> list:
    """Run FileSource → ClassifierNode(whole-image) → Collector and return capture log."""
    src = _make_source(str(VIDEO))
    cfg = vp.ClassifierConfig()
    cfg.input_width = 224
    cfg.input_height = 224
    cfg.target_classes = []
    engine = vp.TrtModelEngine(str(engine_path))
    classifier = vp.ClassifierNode(engine, cfg, "cls")
    collector = _Collector("tail")

    pipeline = vp.Pipeline(vp.PipelineConfig())
    pipeline.add_node(src)
    pipeline.add_node(classifier)
    pipeline.add_node(collector._cpp_node)
    pipeline.connect(src, classifier)
    pipeline.connect(classifier, collector._cpp_node)

    _run_pipeline_briefly(pipeline, run_seconds=run_seconds)
    return collector.captured


def test_whole_image_classify_efficientnet_produces_real_logits():
    _require(VIDEO, EFFNET_ENGINE)
    captured = _run_whole_image_classify(EFFNET_ENGINE)
    assert len(captured) > 0, "no frames flowed through pipeline"
    with_cls = [c for c in captured if c["n_cls"] > 0]
    assert len(with_cls) > 0, (
        f"EfficientNet-B0 produced 0 classifications across {len(captured)} frames"
    )
    det_idx, cid, conf = with_cls[0]["cls"][0]
    assert det_idx == -1
    assert 0 <= cid < 1000, f"class_id={cid} out of [0,1000)"
    assert 0.0 <= conf <= 1.0


def test_whole_image_classify_shufflenet_produces_real_logits():
    _require(VIDEO, SHUFFLE_ENGINE)
    captured = _run_whole_image_classify(SHUFFLE_ENGINE)
    assert len(captured) > 0
    with_cls = [c for c in captured if c["n_cls"] > 0]
    assert len(with_cls) > 0, (
        f"ShuffleNetV2 produced 0 classifications across {len(captured)} frames"
    )
    det_idx, cid, conf = with_cls[0]["cls"][0]
    assert det_idx == -1
    assert 0 <= cid < 1000
    assert 0.0 <= conf <= 1.0


# --- 5) YOLOv8m-seg real instance segmentation ---

def test_yolov8m_seg_real_inference_produces_detections_and_masks():
    """Real YOLOv8m-seg: FileSource → YoloSegNode → Collector.
    Asserts both frame.detections (with valid bbox/conf/class) and frame.masks populate.
    """
    _require(VIDEO, SEG_ENGINE)

    src = _make_source(str(VIDEO))

    seg_cfg = vp.YoloSegConfig()
    seg_cfg.input_width = 640
    seg_cfg.input_height = 640
    seg_cfg.score_threshold = 0.25
    seg_cfg.nms_threshold = 0.45
    seg_cfg.mask_threshold = 0.5
    seg_engine = vp.TrtModelEngine(str(SEG_ENGINE))
    seg = vp.YoloSegNode(seg_engine, seg_cfg, "seg")

    collector = _Collector("tail")

    pipeline = vp.Pipeline(vp.PipelineConfig())
    pipeline.add_node(src)
    pipeline.add_node(seg)
    pipeline.add_node(collector._cpp_node)
    pipeline.connect(src, seg)
    pipeline.connect(seg, collector._cpp_node)

    _run_pipeline_briefly(pipeline, run_seconds=20.0)

    captured = collector.captured
    assert len(captured) > 0, "no frames flowed through seg pipeline"

    with_det = [c for c in captured if c["n_det"] > 0]
    assert len(with_det) > 0, (
        f"YOLOv8m-seg produced 0 detections across {len(captured)} frames"
    )

    # 同时验证 masks: YOLOv8-seg 应该每个 detection 都对应一个 mask
    with_masks = [c for c in captured if c["n_masks"] > 0]
    assert len(with_masks) > 0, (
        f"YOLOv8m-seg produced 0 masks across {len(captured)} frames"
    )

    sample = with_masks[0]
    assert sample["n_masks"] == sample["n_det"], (
        f"expected n_masks == n_det, got n_masks={sample['n_masks']} n_det={sample['n_det']}"
    )

    # 检测合法性
    for cid, score in zip(sample["det_classes"], sample["det_scores"]):
        assert cid >= 0, f"class_id should be ≥0, got {cid}"
        assert cid < 80, f"COCO has 80 classes, got class_id={cid}"
        assert 0.25 <= score <= 1.0, f"confidence={score} out of [score_threshold, 1]"
