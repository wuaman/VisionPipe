# VisionPipe-py

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**VisionPipe-py** 是一个面向生产环境的**视频 AI 推理框架**，底层由 C++（CUDA/TensorRT）驱动以保证高性能，业务层由 Python 实现以保证灵活性。框架以**有向无环图（DAG）节点管道**为核心抽象，用户通过 Python DSL 编排节点，框架负责调度、并发、资源管理和硬件适配。

## 核心特点

| 特点 | 说明 |
|------|------|
| **Python DSL 编排** | 用 `>>` 运算符连接节点构图，可导出/导入 YAML 用于版本化和运维下发 |
| **C++ 热路径，零 GIL 干扰** | 推理、编解码、调度全在 C++ 线程池；Python 业务节点回调时短暂 acquire GIL |
| **同进程多 Pipeline** | `PipelineManager` 支持动态创建/销毁多条 pipeline，无需 Docker 隔离 |
| **模型去重复用** | `ModelRegistry` 按引擎文件 SHA-256 去重，多条 pipeline 共享同一 `IModelEngine`，节省显存 |
| **优雅启停协议** | DRAINING → teardown → STOPPED 三段式退出，GPU 资源安全释放（<500ms） |
| **节点并发扩展** | 瓶颈节点配置 `parallel_workers=N`，多个 worker 共享模型权重独立执行上下文 |
| **有界队列 + 溢出策略** | 每节点有界输入队列；实时流默认 `DROP_OLDEST` 保低延迟，文件处理可选 `BLOCK` 不丢帧 |
| **ROI 实时热更** | 前端 canvas 框选 → WebSocket 归一化坐标 → C++ `set_param()` 原子写 → 下一帧生效 |
| **HAL 硬件抽象** | `IModelEngine` / `IExecContext` / `IAllocator` 三接口屏蔽厂商差异 |
| **内置可观测性** | 每节点暴露队列占用率、丢帧计数、FPS；健康接口 `GET /pipelines/{id}/health` |

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
│  │  IModelEngine   IExecContext   IAllocator   ICodec (二期)       │  │
│  │       │               │            │           │               │  │
│  │  TrtEngine     TrtExecCtx    CudaAlloc    NvDecCodec (二期)    │  │
│  │  AscendEngine  AscendExecCtx AclAlloc     AscendCodec (三期)   │  │
│  │  RknnEngine    RknnExecCtx   RknnAlloc    (四期)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 环境要求

### 必需依赖

| 组件 | 版本要求 | 验证命令 |
|------|----------|----------|
| CUDA Toolkit | >=11.8 | `nvcc --version` |
| cuDNN | >=8.6 | `cat /usr/local/cuda/include/cudnn_version.h` |
| TensorRT | >=8.6 | `trtexec --version` |
| Python | >=3.10 | `python3 --version` |
| CMake | >=3.20 | `cmake --version` |
| GCC | >=9.0 | `g++ --version` |
| uv | 最新版 | `uv --version` |
| clang-format | >=14 | `clang-format --version` |

### GPU 环境

⚠️ **重要**：本项目的所有测试（单元测试、集成测试、E2E 测试）均需在真实 GPU 环境运行。

- 开发机必须配备 NVIDIA GPU（推荐 RTX 3090 或更高）
- CUDA 驱动版本 >= 525.60.13
- 确保 `nvidia-smi` 正常输出

## 快速开始

### 1. 安装 uv（Python 包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆项目并创建虚拟环境

```bash
git clone https://github.com/your-org/VisionPipe-py.git
cd VisionPipe-py

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

uv pip install -e ".[dev]"
```

### 3. 构建 C++ 核心库和 Python 扩展

```bash
# 配置并构建
cmake -B build
cmake --build build

# 构建 Python 扩展（nanobind）
cmake --build build --target visionpipe_python
```

### 4. 运行测试

```bash
# C++ 测试（需要 GPU）
ctest --test-dir build

# Python 测试（需要 GPU）
uv run pytest
```

## 使用示例

> 📁 完整可运行示例在 [`examples/`](examples/) 目录：
> - [`examples/quickstart.py`](examples/quickstart.py) — 单 Pipeline 入门 demo（检测 + JSON 输出）
> - [`examples/multi_pipeline_demo.py`](examples/multi_pipeline_demo.py) — 多 Pipeline 并发 + 共享 engine + 生命周期隔离

按 README 完成依赖安装与构建后，10 分钟内即可跑通：

```bash
# 1. 单 Pipeline 检测演示（运行 5 秒，打印前 3 帧 JSON 结果与 FPS）
uv run python examples/quickstart.py

# 2. 多 Pipeline 并发演示（vehicle vs person 双场景，验证类别隔离 + 生命周期隔离）
uv run python examples/multi_pipeline_demo.py
```

完整 Python API 参考见 [`docs/api_reference.md`](docs/api_reference.md)。

### Python DSL 编排 Pipeline

```python
from visionpipe import (
    FileSource, SourceConfig, DecodeMode,
    DetectorNode, DetectorConfig, TrtModelEngine,
    JsonResultSink, JsonResultSinkConfig,
)

# 定义节点
src_cfg = SourceConfig("video.mp4")
src_cfg.decode_mode = DecodeMode.AUTO  # NVDEC 优先，不可用则回退 CPU
src = FileSource(src_cfg)

engine = TrtModelEngine("models/yolov8/yolov8n_dynamic.engine")
det_cfg = DetectorConfig()
det_cfg.score_threshold = 0.25
det_cfg.workers = 2  # 并行 worker，自动按 frame_id 重排序输出
det = DetectorNode(engine, det_cfg, "detector")

sink = JsonResultSink(JsonResultSinkConfig(), "sink")

# 用 >> 运算符链式构建 Pipeline (Phase 3 DSL)
pipe = src >> det >> sink

# 启动 (block=True 阻塞至 source 自然结束；block=False 后台运行)
pipe.run(block=False)
# ... 业务消费 sink.pop_json(timeout_ms=200) ...
pipe.stop()
```

### 自定义业务节点

VisionPipe-py 提供两种自定义节点：

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

# 链路：PyNode/CustomNode 自动 unwrap 到 _cpp_node
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
# 退出前调用 node.stop() 释放子进程
```

### YAML 配置导入/导出

```python
# 导出 pipeline 拓扑 + 节点参数到 YAML
pipe.export_yaml("pipeline.yaml")

# 仅解析 YAML 得到 PipelineSpec（不需要 GPU）
spec = Pipeline.load_yaml("pipeline.yaml")

# 完整重建：需用 node_overrides 注入有外部依赖的节点
rebuilt = Pipeline.from_yaml(
    "pipeline.yaml",
    node_overrides={"src": src, "det": det, "sink": sink},
)
```

### REST 管理 API

启动嵌入式 REST + WebSocket 服务（aiohttp）：

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

常用端点：

```bash
# 创建 pipeline (body: {"spec": <PipelineSpec dict or YAML string>})
curl -X POST http://localhost:8080/pipelines -H "Content-Type: application/json" -d @spec.json

# 列表 / 启动 / 停止 / 销毁
curl http://localhost:8080/pipelines
curl -X POST http://localhost:8080/pipelines/{id}/start
curl -X POST http://localhost:8080/pipelines/{id}/stop
curl -X DELETE http://localhost:8080/pipelines/{id}

# 健康 + 节点状态
curl http://localhost:8080/pipelines/{id}/health
curl http://localhost:8080/pipelines/{id}/nodes

# 通用参数热更
curl -X POST http://localhost:8080/pipelines/{id}/params \
  -H "Content-Type: application/json" \
  -d '{"node_id": "detector", "param_name": "score_threshold", "value": 0.5}'
```

WebSocket 端点：

| 路径 | 用途 |
|------|------|
| `/ws/{id}/results` | 推送 JsonResultSink 每帧 JSON |
| `/ws/{id}/control` | 通用控制通道（`ping` / `set_param` / `roi`） |
| `/ws/{id}/webrtc` | WebRTC SDP/ICE 信令（需启用 `VISIONPIPE_USE_WEBRTC=ON`） |

## 项目结构

```
VisionPipe-py/
├── src/
│   ├── core/                # C++ 核心调度框架
│   │   ├── pipeline.h/cpp           # Pipeline 执行引擎
│   │   ├── pipeline_manager.h/cpp   # Pipeline 生命周期管理
│   │   ├── node_base.h/cpp          # 节点基类
│   │   ├── infer_node.h/cpp         # 推理节点（parallel_workers）
│   │   ├── model_registry.h/cpp     # 模型去重、引用计数、TTL
│   │   ├── bounded_queue.h          # 有界队列（DROP_OLDEST/BLOCK）
│   │   ├── frame.h                  # Frame 数据结构
│   │   ├── tensor.h                 # Tensor 内存管理
│   │   └── error.h                  # 异常层次
│   │
│   ├── hal/                 # 硬件抽象层
│   │   ├── imodel_engine.h          # IModelEngine 接口
│   │   └── nvidia/                  # NVIDIA TensorRT 实现（一期）
│   │
│   └── nodes/               # 节点实现
│       ├── source/                  # FileSource / RtspSource
│       ├── infer/                   # DetectorNode / ClassifierNode / SegmentNode
│       └── sink/                    # WebRTCSink / JsonResultSink / MjpegSink
│
├── python/
│   ├── visionpipe/          # Python 包
│   ├── bindings/            # nanobind C++ 绑定
│   └── server/              # REST API + WebSocket
│
├── tests/
│   ├── unit/cpp/            # C++ 单元测试（Google Test）
│   ├── unit/python/         # Python 单元测试（pytest）
│   ├── integration/         # 集成测试（需 GPU）
│   └── e2e/                 # 端到端测试
│
├── models/                  # 模型文件（ONNX/TRT Engine）
├── benchmarks/              # 性能基准测试
├── examples/                # 示例代码
├── docs/                    # 文档
│
├── CMakeLists.txt           # CMake 配置
├── pyproject.toml           # Python 包配置（uv）
├── CLAUDE.md                # Claude Code 开发指南
├── DEV_SPEC.md              # 详细开发规范
└── README.md                # 本文件
```

## 开发路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| Phase 0 | 工程骨架 + CI 基础 | ✅ 完成 |
| Phase 1 | C++ 核心调度框架 | ✅ 完成 |
| Phase 2 | NVIDIA 推理 + 编解码 | ✅ 完成 |
| Phase 3 | Python 绑定 + DSL | ✅ 完成 |
| Phase 4 | 管理 API + 前端交付 | ✅ 完成 |
| Phase 5 | 集成验证 + 收尾 | ✅ 完成 |

### 一期验证模型

| 任务 | 模型 | 优先级 |
|------|------|--------|
| 目标检测 | YOLOv8 / YOLOv11 | P0 |
| 图像分类 | ResNet50 / EfficientNet-B0 / ShuffleNetV2 | P0 |
| 实例分割 | YOLOv8-seg | P1 |
| 目标追踪 | ByteTrack | P1 |

## 性能目标

| 指标 | 目标值（RTX 3090） |
|------|-------------------|
| 单路 1080p YOLOv8 吞吐 | ≥25 FPS |
| 16路 1080p 同卡总吞吐 | ≥200 FPS |
| Pipeline 启动耗时（模型已缓存） | <500ms |
| 优雅停止耗时 | <500ms |
| ROI 热更生效延迟 | ≤1 帧（@25fps = 40ms） |
| GPU 显存占用（16路，共享模型） | 对比不共享减少 ≥30% |

## 开发指南

### 代码风格

```bash
# C++ 格式化
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i

# Python 代码检查和类型检查
uv run ruff check python/
uv run ruff format python/
uv run mypy python/
```

### 运行测试

```bash
# C++ 测试
ctest --test-dir build

# 指定测试
ctest --test-dir build -R test_bounded_queue

# Python 测试
uv run pytest

# 指定测试文件
uv run pytest tests/unit/python/test_bindings.py -v
```

### 阶段门禁

每个开发阶段结束前必须满足：
1. 所有测试通过：`ctest --test-dir build` 和 `uv run pytest`
2. 核心模块覆盖率 >90%，整体 >80%
3. 代码风格检查通过
4. 无内存泄漏（Valgrind/ASAN）

## 包管理策略

| 依赖类型 | 管理方式 | 示例 |
|----------|----------|------|
| Python 包 | uv (pyproject.toml) | pytest, ruff, mypy |
| C++ 重依赖 | 系统包管理器或源码编译 | CUDA, TensorRT, OpenCV（需 CUDA 模块） |
| C++ 轻依赖 | CMake FetchContent | spdlog, nlohmann-json, googletest, nanobind |

## 文档

- [`DEV_SPEC.md`](DEV_SPEC.md) — 详细开发规范与任务清单
- [`CLAUDE.md`](CLAUDE.md) — Claude Code 开发指南
- [`docs/api_reference.md`](docs/api_reference.md) — Python API 完整参考
- [`examples/`](examples/) — 可运行示例代码

## License

Apache License 2.0

## 贡献

欢迎贡献代码、报告问题或提出建议。请遵循：
1. 通过 GitHub Issues 报告问题
2. 提交 PR 前确保所有测试通过
3. 遵循项目的代码风格规范

---

**VisionPipe-py** - 高性能视频 AI 推理框架，让 Python 开发者也能轻松驾驭 GPU 加速的视频处理。