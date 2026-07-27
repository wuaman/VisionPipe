<p align="center">
  <a href="README.md">简体中文</a> · <a href="README_EN.md">English</a>
</p>

# VisionPipe-py

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

> **Write business logic in Python, squeeze the GPU in C++.**
> A production-grade video AI inference framework — built around a DAG node-pipeline abstraction, a single `>>` chain turns decode, inference, tracking and sink into a high-throughput pipeline.

VisionPipe-py hides TensorRT's raw performance inside the C++ hot path and hands the orchestration freedom to Python. You declare nodes and topology; the framework handles scheduling, concurrency, memory reuse, graceful shutdown and hardware adaptation — from a single 1080p real-time stream to 16 concurrent streams on one card, same API, same codebase.

## Why VisionPipe-py

```python
src  >> det >> track >> annotator >> sink     # one line, full GPU-accelerated chain
```

| | Traditional | VisionPipe-py |
|---|---|---|
| **Performance** | Python scheduling + frequent GIL contention | C++ thread pool runs the hot path; Python briefly acquires GIL only on callbacks |
| **VRAM** | One model copy per stream | `ModelRegistry` dedups by SHA-256; multiple streams share one `IModelEngine` — 16 streams save ≥30% VRAM |
| **Orchestration** | Hand-rolled threads/queues/sync | DAG nodes + bounded queues, `>>` operator to compose graphs, YAML serializable |
| **Concurrency** | Single-thread bottleneck or DIY worker pool | Bottleneck nodes set `parallel_workers=N`; workers share weights, own contexts, auto-reorder by `frame_id` |
| **Ops** | Docker isolation per stream | In-process `PipelineManager` spins pipelines up/down dynamically; REST/WS hot-reload params |
| **Shutdown** | Hard kill leaks VRAM | DRAINING → teardown → STOPPED three-phase, safe release in <500ms |

## Core Features

- **🐍 Python DSL orchestration** — chain nodes with the `>>` operator; export/import YAML for versioning and ops deployment
- **⚙️ C++ hot path, zero GIL interference** — inference, codec and scheduling run in C++ thread pools; GIL is acquired briefly only during business-node callbacks
- **🔀 In-process multi-pipeline** — `PipelineManager` creates/destroys pipelines dynamically, no Docker isolation needed
- **♻️ Model dedup & reuse** — `ModelRegistry` dedups by engine-file SHA-256; multiple pipelines share one `IModelEngine`, saving VRAM
- **🛑 Graceful start/stop** — DRAINING → teardown → STOPPED three-phase exit; GPU resources released safely in <500ms
- **🚀 Node concurrency scaling** — bottleneck nodes set `parallel_workers=N`; workers share weights with independent exec contexts, output reordered by `frame_id`
- **📦 Bounded queues + overflow policy** — real-time streams default to `DROP_OLDEST` for low latency; file processing can choose `BLOCK` to drop no frames
- **🎯 Real-time ROI hot-reload** — frontend canvas selection → WebSocket normalized coords → C++ `set_param()` atomic write → effective on the next frame
- **🔌 HAL hardware abstraction** — `IModelEngine` / `IExecContext` / `IAllocator` interfaces shield vendor differences; NVIDIA shipped, Ascend/RKNN reserved
- **📊 Built-in observability** — every node exposes queue occupancy, drop count, FPS; `GET /pipelines/{id}/health` for a one-shot check

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python Layer                                  │
│                                                                       │
│  Pipeline DSL          Business Nodes         Management API          │
│  pipe = Pipeline()     class MyNode(PyNode)   GET/POST /pipelines     │
│  src >> det >> biz     def process(frame):    POST /pipelines/{id}/   │
│  pipe.run()              ...                  params                  │
│                                                                       │
│  ────────────────── nanobind binding layer ──────────────────────── │
│                                                                       │
│                        C++ Core Layer                                 │
│                                                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │ PipelineManager │  │  ModelRegistry   │  │   ControlChannel     │ │
│  │                 │  │                  │  │  (WebSocket + REST)  │ │
│  │ Pipeline[id_A]  │  │ sha256 → Engine  │  │  ROI / set_param()   │ │
│  │ Pipeline[id_B]  │  │ refcount + TTL   │  │  pipeline CRUD       │ │
│  └────────┬────────┘  └──────────────────┘  └──────────────────────┘ │
│           │                                                           │
│  ┌────────▼─────────────────────────────────────────────────────┐    │
│  │                    Pipeline (DAG)                             │    │
│  │                                                               │    │
│  │  SourceNode ──▶ [Queue] ──▶ InferNode ──▶ [Queue] ──▶ ...   │    │
│  │     │                          │                              │    │
│  │  FileSource               TrtInferNode                        │    │
│  │  RtspSource               (parallel_workers=N)               │    │
│  │  (DecodeMode:AUTO/GPU/CPU) Worker0: IExecContext+CudaStream   │    │
│  │                           Worker1: IExecContext+CudaStream   │    │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    HAL Hardware Abstraction                    │  │
│  │  IModelEngine   IExecContext   IAllocator                       │  │
│  │       │               │            │                            │  │
│  │  TrtEngine     TrtExecCtx    CudaAlloc      ← shipped (phase 1) │  │
│  │  AscendEngine  AscendExecCtx AclAlloc       ← reserved          │  │
│  │  RknnEngine    RknnExecCtx   RknnAlloc      ← reserved          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Node Library

| Category | Node | Capability |
|----------|------|------------|
| **Source** | `FileSource` / `RtspSource` | Local file / RTSP stream; NVDEC GPU decode preferred, falls back to CPU when unavailable |
| **Infer** | `DetectorNode` | YOLOv8/11 detection + NMS, supports ROI hot-reload |
| | `ClassifierNode` | Crop + classification inference (ResNet / EfficientNet / ShuffleNet) |
| | `YoloSegNode` | YOLOv8/11-seg instance segmentation (detection + mask dual output) |
| | `RtmPoseNode` | RTMPose top-down keypoint detection (SimCC decode, depends on upstream boxes) |
| | `YoloPoseNode` | YOLOv8/11-pose single-stage keypoints (box + keypoints in one pass, supports frame-level batch) |
| **Track** | `ByteTrackNode` | CPU multi-object tracking |
| **Viz** | `AnnotatorNode` | Detection box / trajectory / mask visualization |
| **Sink** | `JsonResultSink` | JSON structured output (`pop_json`) |
| | `MjpegSink` | JPEG-encoded streaming (`pop_jpeg`) |
| | `WebRTCSink` | WebRTC real-time streaming (libdatachannel + NVENC, requires compile flag) |
| **Custom** | `PyNode` / `CustomNode` | In-process callback (lightweight) / standalone subprocess (heavy logic, true parallelism, no GIL) |

## Quick Start

### 1. Prepare the environment

```bash
# Required: CUDA >=11.8 / cuDNN >=8.6 / TensorRT >=8.6 / Python >=3.10 / CMake >=3.20 / GCC >=9
# Recommended dev GPU: RTX 3090 or better, driver >=525.60.13

curl -LsSf https://astral.sh/uv/install.sh | sh      # install uv
git clone https://github.com/your-org/VisionPipe-py.git
cd VisionPipe-py
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Build the C++ core and Python extension

```bash
cmake -B build && cmake --build build
cmake --build build --target visionpipe_python    # nanobind extension

# Enable WebRTC streaming (optional)
cmake -B build -DVISIONPIPE_USE_WEBRTC=ON && cmake --build build
```

### 3. Run an example

```bash
uv run python examples/quickstart.py    # single pipeline: detection + JSON output, results in 5s
```

### 4. Run tests (GPU required)

```bash
ctest --test-dir build                  # C++ tests
uv run pytest                           # Python tests
```

## Examples

> 📁 Runnable examples in [`examples/`](examples/):
> [`quickstart.py`](examples/quickstart.py) · [`multi_pipeline_demo.py`](examples/multi_pipeline_demo.py) · [`pose_demo.py`](examples/pose_demo.py) · [`segment_demo.py`](examples/segment_demo.py) · [`roi_hotupdate_demo.py`](examples/roi_hotupdate_demo.py) · [`webrtc_demo.py`](examples/webrtc_demo.py)

### Compose a pipeline with the Python DSL

```python
from visionpipe import (
    FileSource, SourceConfig, DecodeMode,
    DetectorNode, DetectorConfig, TrtModelEngine,
    JsonResultSink, JsonResultSinkConfig,
)

src_cfg = SourceConfig("video.mp4")
src_cfg.decode_mode = DecodeMode.AUTO      # NVDEC preferred, falls back to CPU
src = FileSource(src_cfg)

engine = TrtModelEngine("models/yolov8/yolov8n_dynamic.engine")
det_cfg = DetectorConfig()
det_cfg.score_threshold = 0.25
det_cfg.workers = 2                        # parallel workers, auto-reordered by frame_id
det = DetectorNode(engine, det_cfg, "detector")

sink = JsonResultSink(JsonResultSinkConfig(), "sink")

# chain the whole pipeline in one line
pipe = src >> det >> sink
pipe.run(block=False)                      # block=False runs in background
# ... consume sink.pop_json(timeout_ms=200) ...
pipe.stop()
```

### Custom business nodes

```python
# Option 1: PyNode — in-process callback, for lightweight logic
from visionpipe import PyNode

class AlertNode(PyNode):
    def __init__(self, target_classes: list[int]) -> None:
        self._targets = set(target_classes)
        super().__init__(name="alert")

    def process(self, frame) -> None:
        hits = [d for d in frame.detections if d.class_id in self._targets]
        if hits:
            frame.set_user_data("alert_count", len(hits))

pipe = src >> det >> AlertNode([0, 2]) >> sink
```

```python
# Option 2: CustomNode — standalone subprocess, true parallelism with no GIL, for heavy logic
from visionpipe import CustomNode, FrameView

class AnalyzeNode(CustomNode):
    def on_frame(self, frame: FrameView) -> None:
        frame.user_data["analysis"] = my_heavy_compute(frame.detections)

node = AnalyzeNode(name="analyze", process_mode="subprocess")
pipe = src >> det >> node._cpp_node >> sink
node.stop()                                # release subprocess before exit
```

### YAML config import/export

```python
pipe.export_yaml("pipeline.yaml")          # export topology + node params

spec = Pipeline.load_yaml("pipeline.yaml") # parse only (no GPU needed)

rebuilt = Pipeline.from_yaml(              # full rebuild: inject nodes with external deps
    "pipeline.yaml",
    node_overrides={"src": src, "det": det, "sink": sink},
)
```

### REST + WebSocket management API

```python
import asyncio, visionpipe as vp
from visionpipe.server.management_api import ManagementServer

async def main() -> None:
    manager = vp.PipelineManager()
    server = ManagementServer(manager, host="0.0.0.0", port=8080)
    await server.start()
    # ... business running ...
    await server.stop()

asyncio.run(main())
```

```bash
curl -X POST http://localhost:8080/pipelines -H "Content-Type: application/json" -d @spec.json  # create
curl http://localhost:8080/pipelines                                                         # list
curl -X DELETE http://localhost:8080/pipelines/{id}                                          # destroy
curl http://localhost:8080/pipelines/{id}/health                                             # health
curl -X POST http://localhost:8080/pipelines/{id}/params \                                   # hot-reload param
  -H "Content-Type: application/json" \
  -d '{"node_id": "detector", "param_name": "score_threshold", "value": 0.5}'
```

| WS endpoint | Purpose |
|-------------|---------|
| `/ws/{id}/results` | Pushes per-frame JSON from JsonResultSink |
| `/ws/{id}/control` | Generic control channel (`ping` / `set_param` / `roi`) |
| `/ws/{id}/webrtc` | WebRTC SDP/ICE signaling (requires `VISIONPIPE_USE_WEBRTC=ON`) |

## Performance Targets

| Metric | Target (RTX 3090) |
|--------|--------------------|
| Single 1080p YOLOv8 throughput | ≥25 FPS |
| 16× 1080p same-card total throughput | ≥200 FPS |
| Pipeline startup (model cached) | <500ms |
| Graceful shutdown | <500ms |
| ROI hot-reload latency | ≤1 frame (@25fps ≈ 40ms) |
| VRAM usage (16 streams, shared model) | ≥30% reduction vs. unshared |

## Project Structure

```
VisionPipe-py/
├── src/
│   ├── core/          # C++ core scheduling (Pipeline / PipelineManager / InferNode / ModelRegistry / BoundedQueue)
│   ├── hal/           # Hardware abstraction (IModelEngine / IExecContext / IAllocator, NVIDIA impl)
│   └── nodes/         # Node implementations (source / infer / tracker / visualize / sink)
├── python/
│   ├── visionpipe/    # Python package (DSL / PyNode / CustomNode / serialization)
│   ├── bindings/      # nanobind C++ bindings
│   └── server/        # REST API + WebSocket
├── tests/             # unit / integration / e2e (all require GPU)
├── examples/          # runnable examples
├── docs/              # documentation
├── CMakeLists.txt
└── pyproject.toml
```

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| Phase 0 | Project scaffold + CI | ✅ Done |
| Phase 1 | C++ core scheduling framework | ✅ Done |
| Phase 2 | NVIDIA inference + codec | ✅ Core usable |
| Phase 3 | Python bindings + DSL | ✅ Basic done |
| Phase 4 | Management API + frontend delivery | ✅ REST+WS framework done |
| Phase 5 | Integration validation + wrap-up | 🚧 In progress |

**Verified model matrix:** YOLOv8/11 (detection) · ResNet50 / EfficientNet-B0 / ShuffleNetV2 (classification) · YOLOv8-seg (segmentation) · RTMPose / YOLOv8-pose (keypoints) · ByteTrack (tracking)

## Development Guide

```bash
# Code style
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i   # C++
uv run ruff check python/ && uv run ruff format python/         # Python
uv run mypy python/                                             # type check

# Run tests
ctest --test-dir build -R test_bounded_queue                    # specific C++ test
uv run pytest tests/unit/python/test_bindings.py -v             # specific Python test
```

**Phase gate criteria:** all tests pass · core module coverage >90% / overall >80% · style checks pass · no memory leaks (Valgrind/ASAN)

## Documentation

- [`DEV_SPEC.md`](DEV_SPEC.md) — detailed dev spec and task list
- [`CLAUDE.md`](CLAUDE.md) — Claude Code dev guide
- [`docs/api_reference.md`](docs/api_reference.md) — full Python API reference
- [`examples/`](examples/) — runnable example code

## License

Apache License 2.0

## Contributing

Contributions, issues and suggestions are welcome:

1. Report issues via GitHub Issues
2. Ensure all tests pass before submitting a PR (`ctest` + `uv run pytest`)
3. Follow the project's code style

---

**VisionPipe-py** — let Python developers harness GPU-accelerated video processing with ease.
