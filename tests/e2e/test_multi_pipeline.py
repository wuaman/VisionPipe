"""E2E tests for T5.1: Multi-Pipeline Concurrent Integration.

Tests verify:
  1. Two pipelines run concurrently and produce disjoint class-ID sets in their
     results (simulating a "physics lab" pipeline vs a "chemistry lab" pipeline).
  2. When both pipelines share the same TensorRT engine instance, VRAM increment
     for the second pipeline is ≤ 10% of single-pipeline VRAM usage.

Requires: NVIDIA GPU (CUDA). All GPU-dependent tests are skipped when no GPU is
detected.  The VRAM-sharing test additionally requires pynvml and a TRT engine.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe
from visionpipe.py_node import PyNode

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

_GPU_AVAILABLE: bool | None = None


def _has_gpu() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            visionpipe.PipelineManager()
            _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def _trt_engine_path() -> Path | None:
    candidates = [
        ROOT / "tests" / "data" / "yolov8n_dynamic.engine",
        ROOT / "tests" / "data" / "yolov8n_fp16.engine",
    ]
    return next((p for p in candidates if p.exists()), None)


def _has_pynvml() -> bool:
    try:
        import pynvml  # noqa: F401
        return True
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")
requires_trt_nvml = pytest.mark.skipif(
    _trt_engine_path() is None or not _has_pynvml() or not _has_gpu(),
    reason="TRT engine file + pynvml + GPU required",
)

TEST_VIDEO = ROOT / "tests" / "data" / "test.mp4"


# ---------------------------------------------------------------------------
# Synthetic injection node
# ---------------------------------------------------------------------------

class ClassInjectNode(PyNode):
    """PyNode that stamps every frame with synthetic detections at given class IDs.

    Replaces any existing detections so that each downstream sink receives only
    the class IDs configured for this pipeline.
    """

    def __init__(self, class_ids: list[int], name: str = "inject") -> None:
        self._class_ids = list(class_ids)
        super().__init__(name=name)

    def process(self, frame: Any) -> None:
        dets = []
        for cid in self._class_ids:
            det = visionpipe.Detection()
            det.bbox = [0.1, 0.1, 0.9, 0.9]
            det.class_id = cid
            det.confidence = 0.9
            dets.append(det)
        frame.detections = dets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_results(sink: Any, duration_s: float) -> list[dict]:
    """Drain JsonResultSink for *duration_s* seconds and return parsed frames."""
    results: list[dict] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        json_str = sink.pop_json(200)
        if json_str is not None:
            results.append(json.loads(json_str))
    return results


def _build_inject_pipeline(
    video_path: str,
    class_ids: list[int],
    pipeline_name: str,
) -> tuple[Any, ClassInjectNode, Any]:
    """Build a FileSource → ClassInjectNode → JsonResultSink pipeline.

    Returns (pipeline, inject_node, json_sink).  The caller is responsible for
    keeping *inject_node* alive for the duration of the pipeline run.
    """
    source = visionpipe.FileSource(video_path, visionpipe.DecodeMode.CPU)

    inject = ClassInjectNode(class_ids=class_ids, name=f"inject_{pipeline_name}")
    sink_cfg = visionpipe.JsonResultSinkConfig()
    sink_cfg.include_tracks = False
    sink = visionpipe.JsonResultSink(sink_cfg, f"sink_{pipeline_name}")

    cfg = visionpipe.PipelineConfig()
    cfg.name = pipeline_name
    pipeline = visionpipe.Pipeline(cfg)
    pipeline.add_node(source)
    pipeline.add_node(inject._cpp_node)
    pipeline.add_node(sink)
    pipeline.connect(source, inject._cpp_node)
    pipeline.connect(inject._cpp_node, sink)

    return pipeline, inject, sink


# ---------------------------------------------------------------------------
# Test 1: concurrent pipelines produce disjoint class-ID sets
# ---------------------------------------------------------------------------

@requires_gpu
def test_two_pipelines_concurrent_disjoint_classes() -> None:
    """Physics and chemistry pipelines run concurrently; class ID sets must be disjoint.

    Acceptance criteria (T5.1):
      - Both pipelines run simultaneously.
      - Results from each pipeline contain only their designated class IDs.
      - The two class-ID sets are disjoint (no cross-contamination).
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    PHYSICS_IDS = [0, 1, 2]
    CHEMISTRY_IDS = [10, 11, 12]
    video = str(TEST_VIDEO)

    manager = visionpipe.PipelineManager()

    phys_pipeline, phys_inject, phys_sink = _build_inject_pipeline(
        video, PHYSICS_IDS, "physics"
    )
    chem_pipeline, chem_inject, chem_sink = _build_inject_pipeline(
        video, CHEMISTRY_IDS, "chemistry"
    )

    phys_id = manager.create_pipeline(phys_pipeline)
    chem_id = manager.create_pipeline(chem_pipeline)
    manager.start(phys_id)
    manager.start(chem_id)

    phys_results: list[dict] = []
    chem_results: list[dict] = []
    errors: list[Exception] = []

    def collect(sink: Any, out: list[dict]) -> None:
        try:
            out.extend(_collect_results(sink, duration_s=3.0))
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=collect, args=(phys_sink, phys_results), daemon=True)
    t2 = threading.Thread(target=collect, args=(chem_sink, chem_results), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    try:
        assert not errors, f"Result collection raised exceptions: {errors}"
        assert phys_results, "Physics pipeline produced no JSON results within 3 s"
        assert chem_results, "Chemistry pipeline produced no JSON results within 3 s"

        phys_class_ids = {
            det["class_id"]
            for r in phys_results
            for det in r.get("detections", [])
        }
        chem_class_ids = {
            det["class_id"]
            for r in chem_results
            for det in r.get("detections", [])
        }

        assert phys_class_ids == set(PHYSICS_IDS), (
            f"Physics class IDs: expected {set(PHYSICS_IDS)}, got {phys_class_ids}"
        )
        assert chem_class_ids == set(CHEMISTRY_IDS), (
            f"Chemistry class IDs: expected {set(CHEMISTRY_IDS)}, got {chem_class_ids}"
        )
        assert phys_class_ids.isdisjoint(chem_class_ids), (
            f"Class ID sets overlap: {phys_class_ids & chem_class_ids}"
        )
    finally:
        for pid in [phys_id, chem_id]:
            try:
                manager.stop(pid)
            except Exception:
                pass
            try:
                manager.destroy(pid)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test 2: two pipelines share the same TRT backbone — VRAM delta ≤ 10%
# ---------------------------------------------------------------------------

@requires_trt_nvml
def test_shared_backbone_vram_increment() -> None:
    """VRAM increment when adding a second pipeline sharing the same TRT engine is ≤ 10%.

    Acceptance criteria (T5.1):
      - Single TrtModelEngine instance shared between two DetectorNodes.
      - VRAM after adding pipeline 2 increases by ≤ 10% of what pipeline 1 consumed.
    """
    import pynvml

    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    engine_path = _trt_engine_path()
    assert engine_path is not None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def used_vram_bytes() -> int:
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used

    manager = visionpipe.PipelineManager()
    # Shared engine — loaded once, referenced by both pipelines
    shared_engine = visionpipe.TrtModelEngine(str(engine_path))

    baseline_vram = used_vram_bytes()

    def _build_detector_pipeline(name: str, engine: Any) -> tuple[str, Any]:
        source = visionpipe.FileSource(str(TEST_VIDEO), visionpipe.DecodeMode.CPU)
        det = visionpipe.DetectorNode(engine, visionpipe.DetectorConfig(), f"det_{name}")
        sink = visionpipe.JsonResultSink(visionpipe.JsonResultSinkConfig(), f"sink_{name}")
        cfg = visionpipe.PipelineConfig()
        cfg.name = name
        pipeline = visionpipe.Pipeline(cfg)
        pipeline.add_node(source).add_node(det).add_node(sink)
        pipeline.connect(source, det)
        pipeline.connect(det, sink)
        pid = manager.create_pipeline(pipeline)
        return pid, sink

    id1, sink1 = _build_detector_pipeline("trt-pipe-1", shared_engine)
    manager.start(id1)
    time.sleep(1.0)
    vram_single = used_vram_bytes() - baseline_vram

    id2, sink2 = _build_detector_pipeline("trt-pipe-2", shared_engine)
    manager.start(id2)
    time.sleep(0.5)
    vram_delta = used_vram_bytes() - baseline_vram - vram_single

    try:
        assert vram_single > 0, (
            "Pipeline 1 consumed no measurable VRAM — TRT engine may not have loaded"
        )
        assert vram_delta <= 0.10 * vram_single, (
            f"VRAM delta {vram_delta / 1024**2:.1f} MiB exceeds 10% of single-pipeline "
            f"VRAM {vram_single / 1024**2:.1f} MiB "
            f"(actual delta ratio: {vram_delta * 100 / vram_single:.1f}%)"
        )
    finally:
        for pid in [id1, id2]:
            try:
                manager.stop(pid)
            except Exception:
                pass
            try:
                manager.destroy(pid)
            except Exception:
                pass
        pynvml.nvmlShutdown()


# ---------------------------------------------------------------------------
# Test 3: pipeline manager reports correct state transitions under concurrency
# ---------------------------------------------------------------------------

@requires_gpu
def test_concurrent_pipeline_lifecycle_states() -> None:
    """PipelineManager correctly tracks RUNNING state for two concurrent pipelines."""
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    video = str(TEST_VIDEO)
    manager = visionpipe.PipelineManager()

    pipe1, inject1, sink1 = _build_inject_pipeline(video, [0], "state-pipe-1")
    pipe2, inject2, sink2 = _build_inject_pipeline(video, [1], "state-pipe-2")

    id1 = manager.create_pipeline(pipe1)
    id2 = manager.create_pipeline(pipe2)

    assert manager.status(id1) == visionpipe.PipelineStatus.INIT
    assert manager.status(id2) == visionpipe.PipelineStatus.INIT

    manager.start(id1)
    manager.start(id2)
    time.sleep(0.3)

    assert manager.status(id1) == visionpipe.PipelineStatus.RUNNING
    assert manager.status(id2) == visionpipe.PipelineStatus.RUNNING
    assert set(manager.list()) >= {id1, id2}

    try:
        manager.stop(id1)
        manager.stop(id2)
        time.sleep(0.2)
        assert manager.status(id1) == visionpipe.PipelineStatus.STOPPED
        assert manager.status(id2) == visionpipe.PipelineStatus.STOPPED
    finally:
        for pid in [id1, id2]:
            try:
                manager.destroy(pid)
            except Exception:
                pass
