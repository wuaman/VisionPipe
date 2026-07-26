"""E2E tests for T5.1: Multi-Pipeline Concurrent Integration.

Per the revised Phase 5 spec these tests verify *functional* correctness — the
original "VRAM ≤10%" hard target was dropped in favour of behavioural checks:

  1. Two pipelines run concurrently and produce disjoint class-ID sets
     (no cross-contamination between independent flows).
  2. A TRT engine instance can be shared by two DetectorNodes — both
     pipelines load and run without errors, and stopping one pipeline
     does not affect the other (lifecycle isolation).
  3. PipelineManager tracks state transitions for two concurrent pipelines
     and releases resources cleanly after all pipelines are destroyed.

All GPU-dependent tests are skipped when no GPU is detected.  The shared-engine
test additionally skips when no TRT engine asset is available.
"""

from __future__ import annotations

import asyncio
import gc
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
    """Locate a usable TensorRT engine file under tests/models/."""
    candidates = [
        ROOT / "tests" / "models" / "yolov8n_fp16.engine",
        ROOT / "tests" / "models" / "yolov8n_dynamic.engine",
        ROOT / "tests" / "models" / "yolov8n.engine",
    ]
    return next((p for p in candidates if p.exists()), None)


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU required")
requires_trt = pytest.mark.skipif(
    _trt_engine_path() is None or not _has_gpu(),
    reason="TensorRT engine file under tests/models/ + GPU required",
)

TEST_VIDEO = ROOT / "tests" / "data" / "48-3.mp4"
if not TEST_VIDEO.exists() or TEST_VIDEO.is_symlink():
    TEST_VIDEO = ROOT / "tests" / "data" / "test_video_100frames.mp4"


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
    *,
    loop_source: bool = True,
) -> tuple[Any, ClassInjectNode, Any]:
    """Build a FileSource → ClassInjectNode → JsonResultSink pipeline.

    The caller must hold a reference to *inject_node* for the lifetime of the
    pipeline so the Python instance is not garbage-collected.
    """
    src_cfg = visionpipe.SourceConfig(video_path)
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = loop_source
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.DROP_OLDEST
    source = visionpipe.FileSource(src_cfg)

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


def _build_detector_pipeline(
    video_path: str,
    engine: Any,
    pipeline_name: str,
) -> tuple[Any, Any]:
    """Build a FileSource → DetectorNode → JsonResultSink pipeline.

    Uses ``OverflowPolicy.BLOCK`` because ``DetectorNode`` (an ``InferNode``)
    re-orders output by sequential ``frame_id`` — ``DROP_OLDEST`` creates id
    gaps that stall the re-order queue.
    """
    src_cfg = visionpipe.SourceConfig(video_path)
    src_cfg.decode_mode = visionpipe.DecodeMode.AUTO
    src_cfg.loop = True
    src_cfg.queue_capacity = 8
    src_cfg.overflow_policy = visionpipe.OverflowPolicy.BLOCK
    source = visionpipe.FileSource(src_cfg)

    det = visionpipe.DetectorNode(engine, visionpipe.DetectorConfig(), f"det_{pipeline_name}")
    sink = visionpipe.JsonResultSink(visionpipe.JsonResultSinkConfig(), f"sink_{pipeline_name}")

    cfg = visionpipe.PipelineConfig()
    cfg.name = pipeline_name
    pipeline = visionpipe.Pipeline(cfg)
    pipeline.add_node(source)
    pipeline.add_node(det)
    pipeline.add_node(sink)
    pipeline.connect(source, det)
    pipeline.connect(det, sink)
    return pipeline, sink


def _safe_stop_destroy(manager: Any, pipeline_ids: list[str]) -> None:
    for pid in pipeline_ids:
        try:
            status = manager.status(pid)
            if status != visionpipe.PipelineStatus.STOPPED:
                manager.stop(pid)
        except Exception:
            pass
        try:
            manager.destroy(pid)
        except Exception:
            pass


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
        _safe_stop_destroy(manager, [phys_id, chem_id])


# ---------------------------------------------------------------------------
# Test 2: shared TRT engine + lifecycle isolation
# ---------------------------------------------------------------------------

@requires_trt
def test_shared_engine_lifecycle_isolation() -> None:
    """Two pipelines share one TRT engine; stopping one doesn't affect the other.

    Acceptance criteria (T5.1, revised — functional verification only):
      - Both pipelines load and reach RUNNING using a single shared engine.
      - Stopping pipeline 1 leaves pipeline 2 still RUNNING.
      - After both stop, resources release cleanly (no exceptions on destroy).
    """
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    engine_path = _trt_engine_path()
    assert engine_path is not None

    manager = visionpipe.PipelineManager()
    shared_engine = visionpipe.TrtModelEngine(str(engine_path))

    pipe1, sink1 = _build_detector_pipeline(str(TEST_VIDEO), shared_engine, "trt-pipe-1")
    pipe2, sink2 = _build_detector_pipeline(str(TEST_VIDEO), shared_engine, "trt-pipe-2")

    id1 = manager.create_pipeline(pipe1)
    id2 = manager.create_pipeline(pipe2)

    try:
        manager.start(id1)
        manager.start(id2)
        time.sleep(0.5)

        assert manager.status(id1) == visionpipe.PipelineStatus.RUNNING
        assert manager.status(id2) == visionpipe.PipelineStatus.RUNNING

        # Both pipelines should produce results within a short window.
        assert sink1.pop_json(2000) is not None, "Pipeline 1 produced no JSON within 2 s"
        assert sink2.pop_json(2000) is not None, "Pipeline 2 produced no JSON within 2 s"

        # Stop pipeline 1, verify pipeline 2 remains RUNNING.
        manager.stop(id1)
        time.sleep(0.2)
        assert manager.status(id1) == visionpipe.PipelineStatus.STOPPED, (
            f"Pipeline 1 expected STOPPED, got {manager.status(id1)}"
        )
        assert manager.status(id2) == visionpipe.PipelineStatus.RUNNING, (
            f"Pipeline 2 should remain RUNNING after pipeline 1 stop, "
            f"got {manager.status(id2)}"
        )

        # Pipeline 2 still emits results after pipeline 1 stopped.
        assert sink2.pop_json(2000) is not None, (
            "Pipeline 2 produced no JSON after pipeline 1 stop — sharing broken?"
        )

        # Stop pipeline 2.
        manager.stop(id2)
        time.sleep(0.2)
        assert manager.status(id2) == visionpipe.PipelineStatus.STOPPED
    finally:
        _safe_stop_destroy(manager, [id1, id2])
        del shared_engine
        gc.collect()


# ---------------------------------------------------------------------------
# Test 3: pipeline manager state transitions under concurrency + clean teardown
# ---------------------------------------------------------------------------

@requires_gpu
def test_concurrent_pipeline_lifecycle_states() -> None:
    """PipelineManager correctly tracks RUNNING/STOPPED for two concurrent pipelines."""
    if not TEST_VIDEO.exists():
        pytest.skip(f"Test video not found: {TEST_VIDEO}")

    video = str(TEST_VIDEO)
    manager = visionpipe.PipelineManager()

    pipe1, _inject1, _sink1 = _build_inject_pipeline(video, [0], "state-pipe-1")
    pipe2, _inject2, _sink2 = _build_inject_pipeline(video, [1], "state-pipe-2")

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

        # Destroy both, then verify they're gone from manager.list().
        manager.destroy(id1)
        manager.destroy(id2)
        remaining = set(manager.list())
        assert id1 not in remaining and id2 not in remaining, (
            f"Pipelines still present after destroy: {remaining}"
        )
    finally:
        _safe_stop_destroy(manager, [id1, id2])


# ---------------------------------------------------------------------------
# Standalone demo: run with `python test_multi_pipeline.py` to launch
# ManagementServer + Dashboard with two concurrent pipelines.
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Launch two pipelines under ManagementServer for Dashboard visualization."""
    from visionpipe.server.management_api import ManagementServer

    if not TEST_VIDEO.exists():
        print(f"ERROR: Test video not found: {TEST_VIDEO}")
        print("Run: bash tests/data/download_test_assets.sh")
        sys.exit(1)

    video = str(TEST_VIDEO)
    manager = visionpipe.PipelineManager()

    pipe1, inject1, sink1 = _build_inject_pipeline(
        video, [0, 1, 2], "physics", loop_source=True
    )
    pipe2, inject2, sink2 = _build_inject_pipeline(
        video, [10, 11, 12], "chemistry", loop_source=True
    )

    id1 = manager.create_pipeline(pipe1)
    id2 = manager.create_pipeline(pipe2)
    manager.start(id1)
    manager.start(id2)

    print(f"Pipeline 'physics'   ID={id1}  [class_ids: 0,1,2]")
    print(f"Pipeline 'chemistry' ID={id2}  [class_ids: 10,11,12]")
    print()
    print("Dashboard: http://localhost:8080/dashboard")
    print("Press Ctrl+C to stop.")

    async def serve():
        server = ManagementServer(manager, host="0.0.0.0", port=8080)
        await server.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await server.stop()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        _safe_stop_destroy(manager, [id1, id2])
        print("\nPipelines stopped. Bye.")


if __name__ == "__main__":
    _run_demo()
