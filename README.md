<p align="center">
  <a href="README.md">简体中文</a> · <a href="README_EN.md">English</a>
</p>

# VisionPipe-py

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

> **用 Python 写业务，用 C++ 榨干 GPU。**
> 面向生产环境的视频 AI 推理框架 —— 以 DAG 节点管道为核心抽象，一条 `>>` 链就能把解码、推理、跟踪、落库串成高吞吐流水线。

VisionPipe-py 把 TensorRT 的极致性能藏进 C++ 热路径，把编排自由度交给 Python。你只需声明节点和拓扑，框架替你搞定调度、并发、显存复用、优雅启停和硬件适配 —— 从单路 1080p 实时检测到同卡 16 路并发，同一套 API，同一份代码。

## 为什么选 VisionPipe-py

```python
src  >> det >> track >> annotator >> sink     # 一行声明，全链路 GPU 加速
```

| | 传统做法 | VisionPipe-py |
|---|---|---|
| **性能** | Python 调度 + 频繁 GIL 抢占 | C++ 线程池跑热路径，Python 仅在回调时短暂持 GIL |
| **显存** | 每路一份模型权重 | `ModelRegistry` 按 SHA-256 去重，多路共享同一 `IModelEngine`，16 路省 ≥30% 显存 |
| **编排** | 手写线程/队列/同步 | DAG 节点 + 有界队列，`>>` 运算符构图，YAML 可序列化 |
| **并发** | 单线程瓶颈或手撸 worker 池 | 瓶颈节点配 `parallel_workers=N`，多 worker 共享权重独立上下文，按 `frame_id` 自动重排 |
| **运维** | Docker 隔离多路 | 同进程 `PipelineManager` 动态增删 pipeline，REST/WS 热更参数 |
| **退出** | 强杀留显存泄漏 | DRAINING → teardown → STOPPED 三段式，<500ms 安全释放 |

## 核心特性

- **🐍 Python DSL 编排** — `>>` 运算符连接节点构图，可导出/导入 YAML 用于版本化和运维下发
- **⚙️ C++ 热路径，零 GIL 干扰** — 推理、编解码、调度全在 C++ 线程池；业务节点回调时才短暂 acquire GIL
- **🔀 同进程多 Pipeline** — `PipelineManager` 动态创建/销毁多条 pipeline，无需 Docker 隔离
- **♻️ 模型去重复用** — `ModelRegistry` 按引擎文件 SHA-256 去重，多 pipeline 共享 `IModelEngine`，省显存
- **🛑 优雅启停协议** — DRAINING → teardown → STOPPED 三段式退出，GPU 资源 <500ms 安全释放
- **🚀 节点并发扩展** — 瓶颈节点配 `parallel_workers=N`，多 worker 共享权重、独立执行上下文，按 `frame_id` 重排序输出
- **📦 有界队列 + 溢出策略** — 实时流默认 `DROP_OLDEST` 保低延迟，文件处理可选 `BLOCK` 不丢帧
- **🎯 ROI 实时热更** — 前端 canvas 框选 → WebSocket 归一化坐标 → C++ `set_param()` 原子写 → 下一帧生效
- **🔌 HAL 硬件抽象** — `IModelEngine` / `IExecContext` / `IAllocator` 三接口屏蔽厂商差异，NVIDIA 已落地，Ascend/RKNN 预留
- **📊 内置可观测性** — 每节点暴露队列占用率、丢帧计数、FPS，`GET /pipelines/{id}/health` 一键体检

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Python 层                                      │
│                                                                       │
│  Pipeline DSL          Business Nodes         Management API          │
│  pipe = Pipeline()     class MyNode(PyNode)   GET/POST /pipelines     │
│  src >> det >> biz     def process(frame):    POST /pipelines/{id}/   │
│  pipe.run()              ...                  params                  │
│                                                                       │
│  ────────────────── nanobind 绑定层 ──────────────────────────────── │
│                                                                       │
│                        C++ 核心层                                     │
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
│  │                    HAL 硬件抽象层                               │  │
│  │  IModelEngine   IExecContext   IAllocator                       │  │
│  │       │               │            │                            │  │
│  │  TrtEngine     TrtExecCtx    CudaAlloc      ← 一期已落地        │  │
│  │  AscendEngine  AscendExecCtx AclAlloc       ← 预留              │  │
│  │  RknnEngine    RknnExecCtx   RknnAlloc      ← 预留              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 节点库

| 类别 | 节点 | 能力 |
|------|------|------|
| **Source** | `FileSource` / `RtspSource` | 本地文件 / RTSP 流，NVDEC GPU 解码优先，不可用时回退 CPU |
| **Infer** | `DetectorNode` | YOLOv8/11 目标检测 + NMS，支持 ROI 热更 |
| | `ClassifierNode` | 裁切 + 分类推理（ResNet / EfficientNet / ShuffleNet） |
| | `YoloSegNode` | YOLOv8/11-seg 实例分割（检测 + 掩码双输出） |
| | `RtmPoseNode` | RTMPose top-down 关键点检测（SimCC 解码，依赖上游检测框） |
| | `YoloPoseNode` | YOLOv8/11-pose 单阶段关键点（框 + 关键点一次输出，支持帧级 batch） |
| **Track** | `ByteTrackNode` | CPU 多目标跟踪 |
| **Viz** | `AnnotatorNode` | 检测框 / 轨迹 / 掩码可视化标注 |
| **Sink** | `JsonResultSink` | JSON 结构化输出（`pop_json`） |
| | `MjpegSink` | JPEG 编码推流（`pop_jpeg`） |
| | `WebRTCSink` | WebRTC 实时推流（libdatachannel + NVENC，需启用编译选项） |
| **Custom** | `PyNode` / `CustomNode` | 同进程回调（轻量） / 独立子进程（重逻辑，真并行无 GIL） |

## 快速开始

### 1. 准备环境

```bash
# 必需：CUDA >=11.8 / cuDNN >=8.6 / TensorRT >=8.6 / Python >=3.10 / CMake >=3.20 / GCC >=9
# 推荐开发 GPU：RTX 3090 或更高，驱动 >=525.60.13

curl -LsSf https://astral.sh/uv/install.sh | sh      # 安装 uv
git clone https://github.com/your-org/VisionPipe-py.git
cd VisionPipe-py
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. 构建 C++ 核心与 Python 扩展

```bash
cmake -B build && cmake --build build
cmake --build build --target visionpipe_python    # nanobind 扩展

# 启用 WebRTC 推流（可选）
cmake -B build -DVISIONPIPE_USE_WEBRTC=ON && cmake --build build
```

### 3. 跑通示例

```bash
uv run python examples/quickstart.py    # 单 Pipeline：检测 + JSON 输出，5 秒出结果
```

### 4. 验证测试（需 GPU）

```bash
ctest --test-dir build                  # C++ 测试
uv run pytest                           # Python 测试
```

## 使用示例

> 📁 完整可运行示例在 [`examples/`](examples/)：
> [`quickstart.py`](examples/quickstart.py) · [`multi_pipeline_demo.py`](examples/multi_pipeline_demo.py) · [`pose_demo.py`](examples/pose_demo.py) · [`segment_demo.py`](examples/segment_demo.py) · [`roi_hotupdate_demo.py`](examples/roi_hotupdate_demo.py) · [`webrtc_demo.py`](examples/webrtc_demo.py)

### Python DSL 编排 Pipeline

```python
from visionpipe import (
    FileSource, SourceConfig, DecodeMode,
    DetectorNode, DetectorConfig, TrtModelEngine,
    JsonResultSink, JsonResultSinkConfig,
)

src_cfg = SourceConfig("video.mp4")
src_cfg.decode_mode = DecodeMode.AUTO      # NVDEC 优先，不可用则回退 CPU
src = FileSource(src_cfg)

engine = TrtModelEngine("models/yolov8/yolov8n_dynamic.engine")
det_cfg = DetectorConfig()
det_cfg.score_threshold = 0.25
det_cfg.workers = 2                        # 并行 worker，自动按 frame_id 重排序
det = DetectorNode(engine, det_cfg, "detector")

sink = JsonResultSink(JsonResultSinkConfig(), "sink")

# 一行串起整条流水线
pipe = src >> det >> sink
pipe.run(block=False)                      # block=False 后台运行
# ... 业务消费 sink.pop_json(timeout_ms=200) ...
pipe.stop()
```

### 自定义业务节点

```python
# 方式 1: PyNode — 同进程回调，适合极轻量逻辑
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
# 方式 2: CustomNode — 独立子进程，真并行无 GIL，适合重业务逻辑
from visionpipe import CustomNode, FrameView

class AnalyzeNode(CustomNode):
    def on_frame(self, frame: FrameView) -> None:
        frame.user_data["analysis"] = my_heavy_compute(frame.detections)

node = AnalyzeNode(name="analyze", process_mode="subprocess")
pipe = src >> det >> node._cpp_node >> sink
node.stop()                                # 退出前释放子进程
```

### YAML 配置导入/导出

```python
pipe.export_yaml("pipeline.yaml")          # 导出拓扑 + 节点参数

spec = Pipeline.load_yaml("pipeline.yaml") # 仅解析（不需 GPU）

rebuilt = Pipeline.from_yaml(              # 完整重建：注入有外部依赖的节点
    "pipeline.yaml",
    node_overrides={"src": src, "det": det, "sink": sink},
)
```

### REST + WebSocket 管理 API

```python
import asyncio, visionpipe as vp
from visionpipe.server.management_api import ManagementServer

async def main() -> None:
    manager = vp.PipelineManager()
    server = ManagementServer(manager, host="0.0.0.0", port=8080)
    await server.start()
    # ... 业务运行 ...
    await server.stop()

asyncio.run(main())
```

```bash
curl -X POST http://localhost:8080/pipelines -H "Content-Type: application/json" -d @spec.json  # 创建
curl http://localhost:8080/pipelines                                                         # 列表
curl -X DELETE http://localhost:8080/pipelines/{id}                                          # 销毁
curl http://localhost:8080/pipelines/{id}/health                                             # 健康
curl -X POST http://localhost:8080/pipelines/{id}/params \                                   # 参数热更
  -H "Content-Type: application/json" \
  -d '{"node_id": "detector", "param_name": "score_threshold", "value": 0.5}'
```

| WS 端点 | 用途 |
|---------|------|
| `/ws/{id}/results` | 推送 JsonResultSink 每帧 JSON |
| `/ws/{id}/control` | 通用控制通道（`ping` / `set_param` / `roi`） |
| `/ws/{id}/webrtc` | WebRTC SDP/ICE 信令（需启用 `VISIONPIPE_USE_WEBRTC=ON`） |

## 性能目标

| 指标 | 目标值（RTX 3090） |
|------|-------------------|
| 单路 1080p YOLOv8 吞吐 | ≥25 FPS |
| 16 路 1080p 同卡总吞吐 | ≥200 FPS |
| Pipeline 启动耗时（模型已缓存） | <500ms |
| 优雅停止耗时 | <500ms |
| ROI 热更生效延迟 | ≤1 帧（@25fps ≈ 40ms） |
| GPU 显存占用（16 路，共享模型） | 对比不共享减少 ≥30% |

## 项目结构

```
VisionPipe-py/
├── src/
│   ├── core/          # C++ 核心调度（Pipeline / PipelineManager / InferNode / ModelRegistry / BoundedQueue）
│   ├── hal/           # 硬件抽象层（IModelEngine / IExecContext / IAllocator，NVIDIA 实现）
│   └── nodes/         # 节点实现（source / infer / tracker / visualize / sink）
├── python/
│   ├── visionpipe/    # Python 包（DSL / PyNode / CustomNode / 序列化）
│   ├── bindings/      # nanobind C++ 绑定
│   └── server/        # REST API + WebSocket
├── tests/             # unit / integration / e2e（均需 GPU）
├── examples/          # 可运行示例
├── docs/              # 文档
├── CMakeLists.txt
└── pyproject.toml
```

## 开发路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| Phase 0 | 工程骨架 + CI 基础 | ✅ 完成 |
| Phase 1 | C++ 核心调度框架 | ✅ 完成 |
| Phase 2 | NVIDIA 推理 + 编解码 | ✅ 核心可用 |
| Phase 3 | Python 绑定 + DSL | ✅ 基础完成 |
| Phase 4 | 管理 API + 前端交付 | ✅ REST+WS 框架完成 |
| Phase 5 | 集成验证 + 收尾 | 🚧 进行中 |

**已验证模型矩阵：** YOLOv8/11（检测） · ResNet50 / EfficientNet-B0 / ShuffleNetV2（分类） · YOLOv8-seg（分割） · RTMPose / YOLOv8-pose（关键点） · ByteTrack（跟踪）

## 开发指南

```bash
# 代码风格
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i   # C++
uv run ruff check python/ && uv run ruff format python/         # Python
uv run mypy python/                                             # 类型检查

# 运行测试
ctest --test-dir build -R test_bounded_queue                    # 指定 C++ 测试
uv run pytest tests/unit/python/test_bindings.py -v             # 指定 Python 测试
```

**阶段门禁：** 所有测试通过 · 核心模块覆盖 >90% / 整体 >80% · 风格检查通过 · 无内存泄漏（Valgrind/ASAN）

## 文档

- [`DEV_SPEC.md`](DEV_SPEC.md) — 详细开发规范与任务清单
- [`CLAUDE.md`](CLAUDE.md) — Claude Code 开发指南
- [`docs/api_reference.md`](docs/api_reference.md) — Python API 完整参考
- [`examples/`](examples/) — 可运行示例代码

## License

Apache License 2.0

## 贡献

欢迎贡献代码、报告问题或提出建议：

1. 通过 GitHub Issues 报告问题
2. 提交 PR 前确保所有测试通过（`ctest` + `uv run pytest`）
3. 遵循项目的代码风格规范

---

**VisionPipe-py** — 让 Python 开发者也能轻松驾驭 GPU 加速的视频处理。
