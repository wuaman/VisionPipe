"""Real e2e tests for keypoint detection nodes.

- RtmPoseNode: FileSource → DetectorNode(yolov8n) → RtmPoseNode → Collector
- YoloPoseNode: FileSource → YoloPoseNode → Collector

Both load real FP16 TRT engines; skipped when assets are missing.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import visionpipe as vp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "tests" / "models"
DATA_DIR = PROJECT_ROOT / "tests" / "data"

YOLO_ENGINE = MODELS_DIR / "yolov8n_fp16.engine"
RTMPOSE_ENGINE = MODELS_DIR / "rtmpose-m_body7_fp16.engine"
YOLOPOSE_ENGINE = MODELS_DIR / "yolov8n-pose_fp16.engine"
VIDEO = DATA_DIR / "48-3.mp4"


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing test asset: {p}")


def _make_source(uri: str) -> vp.FileSource:
    cfg = vp.SourceConfig(uri)
    cfg.decode_mode = vp.DecodeMode.AUTO
    cfg.loop = False
    cfg.queue_capacity = 8
    cfg.overflow_policy = vp.OverflowPolicy.BLOCK
    return vp.FileSource(cfg)


class _PoseCollector(vp.PyNode):
    def __init__(self, name: str = "collector", max_frames: int = 120):
        super().__init__(name)
        self.max_frames = max_frames
        self.captured: list = []

    def process(self, frame):
        if len(self.captured) < self.max_frames:
            self.captured.append({
                "num_dets": len(frame.detections),
                "poses": [
                    (p.detection_index,
                     [(k.x, k.y, k.score) for k in p.keypoints])
                    for p in frame.poses
                ],
            })
        return frame


def _run(nodes, collector, seconds: float = 20.0) -> None:
    pipe = vp.Pipeline()
    prev = None
    for n in nodes:
        pipe.add_node(n)
        if prev is not None:
            pipe.connect(prev, n)
        prev = n
    pipe.add_node(collector._cpp_node)
    pipe.connect(prev, collector._cpp_node)
    pipe.start()
    deadline = time.time() + seconds
    while time.time() < deadline and len(collector.captured) < collector.max_frames:
        time.sleep(0.2)
    pipe.stop(False)
    pipe.wait_stop()


def _assert_valid_poses(captured: list) -> None:
    frames_with_pose = [c for c in captured if c["poses"]]
    assert frames_with_pose, f"no poses produced across {len(captured)} frames"

    for c in frames_with_pose:
        for det_idx, kpts in c["poses"]:
            assert len(kpts) == 17, f"expected COCO-17 keypoints, got {len(kpts)}"
            for x, y, score in kpts:
                assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "keypoint not normalized"
                assert score >= 0.0


def test_rtmpose_topdown_real_inference_produces_keypoints():
    """Real RTMPose-m: detector person boxes → SimCC decode → frame.poses."""
    _require(YOLO_ENGINE, RTMPOSE_ENGINE, VIDEO)

    det_engine = vp.TrtModelEngine(str(YOLO_ENGINE))
    det_cfg = vp.DetectorConfig()
    det_cfg.score_threshold = 0.35
    detector = vp.DetectorNode(det_engine, det_cfg, "detector")

    pose_engine = vp.TrtModelEngine(str(RTMPOSE_ENGINE))
    pose_cfg = vp.RtmPoseConfig()
    rtmpose = vp.RtmPoseNode(pose_engine, pose_cfg, "rtmpose")

    collector = _PoseCollector()
    _run([_make_source(str(VIDEO)), detector, rtmpose], collector)

    _assert_valid_poses(collector.captured)

    # top-down: 每个 pose 必须关联一个 person detection
    for c in collector.captured:
        for det_idx, _ in c["poses"]:
            assert 0 <= det_idx < c["num_dets"]


def test_yolo_pose_one_stage_real_inference_produces_keypoints():
    """Real YOLOv8n-pose: single-stage detections + keypoints in one pass."""
    _require(YOLOPOSE_ENGINE, VIDEO)

    engine = vp.TrtModelEngine(str(YOLOPOSE_ENGINE))
    cfg = vp.YoloPoseConfig()
    cfg.score_threshold = 0.35
    node = vp.YoloPoseNode(engine, cfg, "yolo_pose")

    collector = _PoseCollector()
    _run([_make_source(str(VIDEO)), node], collector)

    _assert_valid_poses(collector.captured)

    # 单阶段: poses 与 detections 一一对应
    for c in collector.captured:
        assert len(c["poses"]) == c["num_dets"]
        for det_idx, _ in c["poses"]:
            assert 0 <= det_idx < c["num_dets"]
