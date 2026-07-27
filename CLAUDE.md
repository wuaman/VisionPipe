# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

VisionPipe-py 是一个视频 AI 推理框架，核心由 C++（CUDA/TensorRT）驱动，通过 nanobind 提供 Python 绑定。采用基于 DAG 的管道架构进行 GPU 加速视频处理，支持检测、分类、分割、跟踪全链路。

**技术栈：**
- C++17 + CUDA 11.8+ + TensorRT 8.6+
- Python 3.10+ + nanobind 2.0 绑定
- CMake 3.20+ 构建（scikit-build-core 打包）
- uv 管理 Python 包（pyproject.toml）
- 可选：libdatachannel（WebRTC）

## 构建命令

```bash
# 配置并构建 C++（含所有节点库）
cmake -B build && cmake --build build

# 构建 Python 扩展（nanobind）
cmake --build build --target visionpipe_python

# 启用 WebRTC 支持
cmake -B build -DVISIONPIPE_USE_WEBRTC=ON && cmake --build build

# 运行 C++ 测试（需要 GPU）
ctest --test-dir build

# 运行指定 C++ 测试
ctest --test-dir build -R <test_name>
```

## Python 开发

```bash
# 创建虚拟环境并安装依赖
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 运行 Python 测试（需要 GPU）
uv run pytest

# 运行指定测试文件
uv run pytest tests/unit/python/test_bindings.py -v

# 运行集成测试
uv run pytest tests/integration/python/ -v
```

## 代码风格

```bash
# C++ 格式化
find src -name "*.h" -o -name "*.cpp" | xargs clang-format -i

# Python 代码检查和格式化
uv run ruff check python/
uv run ruff format python/
uv run mypy python/
```

## 架构

```
┌───────────────────────────────────────────────────────────────┐
│                        Python 层                               │
│  Pipeline DSL (>> 运算符)  │  PyNode 自定义节点                  │
│  YAML 序列化               │  ManagementServer (aiohttp REST+WS)│
├───────────────────────────────────────────────────────────────┤
│                     nanobind 绑定层                             │
│  python/bindings/: bind_*.cpp → visionpipe_python 扩展模块      │
├───────────────────────────────────────────────────────────────┤
│                       C++ 核心层                               │
│  PipelineManager │ Pipeline (DAG) │ PipelineBuilder            │
│  ModelRegistry   │ NodeBase        │ InferNode (并行推理基类)    │
│  BoundedQueue<T> │ Frame/Tensor    │ Logger (spdlog)            │
├───────────────────────────────────────────────────────────────┤
│                       节点库层                                  │
│  Source: FileSource, RtspSource (cv::cudacodec NVDEC)          │
│  Infer:  DetectorNode, ClassifierNode, YoloSegNode,            │
│          RtmPoseNode, YoloPoseNode                             │
│  Track:  ByteTrackNode                                         │
│  Sink:   JsonResultSink, MjpegSink, WebRTCSink                 │
│  Viz:    AnnotatorNode                                         │
├───────────────────────────────────────────────────────────────┤
│                     HAL 硬件抽象层                              │
│  IModelEngine │ IExecContext │ IAllocator                      │
│  实现: TrtModelEngine, TrtExecContext, CudaAllocator            │
│  测试: MockModelEngine, MockExecContext                         │
└───────────────────────────────────────────────────────────────┘
```

## 核心组件

| 组件 | 位置 | 用途 |
|------|------|------|
| `PipelineManager` | `src/core/pipeline_manager.h` | Pipeline 的创建/启动/停止/销毁/列表/状态查询 |
| `Pipeline` | `src/core/pipeline.h` | DAG 图：add_node/connect/start/stop/wait_stop，含 validate_dag/has_cycle |
| `PipelineBuilder` | `src/core/pipeline_builder.h` | `>>` 运算符链式构建 Pipeline |
| `ModelRegistry` | `src/core/model_registry.h` | 模型去重（SHA-256）、引用计数、TTL 自动清理线程 |
| `BoundedQueue<T>` | `src/core/bounded_queue.h` | 线程安全有界队列，支持 DROP_OLDEST/BLOCK 策略 |
| `NodeBase` | `src/core/node_base.h` | 所有节点基类，NodeState (INIT/RUNNING/DRAINING/STOPPED)，NodeStats |
| `InferNode` | `src/core/infer_node.h` | 推理基类，parallel_workers 并行 + frame_id 排序恢复 |
| `Frame` | `src/core/frame.h` | 帧数据：frame_id, timestamp, gpu_mat, detections, tracks, masks |
| `Tensor` | `src/core/tensor.h` | Move-only RAII 张量，DataType/MemoryType，CpuAllocator |
| `IModelEngine` | `src/hal/imodel_engine.h` | HAL 接口：load_model/create_context |

## 节点目录

| 节点 | 头文件 | 配置类 | 功能 |
|------|--------|--------|------|
| `FileSource` | `src/nodes/source/file_source.h` | `SourceConfig` | 本地视频解码（cv::cudacodec GPU / VideoCapture CPU 自动回退） |
| `RtspSource` | `src/nodes/source/rtsp_source.h` | `SourceConfig` | RTSP 流解码 |
| `DetectorNode` | `src/nodes/infer/detector_node.h` | `DetectorConfig` | YOLOv8 目标检测 + NMS，支持 set_roi/clear_roi 热更新 |
| `ClassifierNode` | `src/nodes/infer/classifier_node.h` | `ClassifierConfig` | 裁切+分类推理 |
| `YoloSegNode` | `src/nodes/infer/yolo_seg_node.h` | `YoloSegConfig` | YOLOv8/11-seg 实例分割（检测+掩码双输出），旧名 SegmentNode 保留为 Python 别名 |
| `RtmPoseNode` | `src/nodes/infer/rtmpose_node.h` | `RtmPoseConfig` | RTMPose top-down 关键点检测（SimCC 解码，依赖上游检测框） |
| `YoloPoseNode` | `src/nodes/infer/yolo_pose_node.h` | `YoloPoseConfig` | YOLOv8/11-pose 单阶段关键点检测（框+关键点一次输出） |
| `ByteTrackNode` | `src/nodes/tracker/bytetrack_node.h` | `ByteTrackConfig` | CPU 多目标跟踪 |
| `AnnotatorNode` | `src/nodes/visualize/annotator_node.h` | `AnnotatorConfig` | 可视化标注（检测框/轨迹/掩码） |
| `JsonResultSink` | `src/nodes/sink/json_result_sink.h` | `JsonResultSinkConfig` | JSON 结构化输出（内部队列 + pop_json） |
| `MjpegSink` | `src/nodes/sink/mjpeg_sink.h` | `MjpegSinkConfig` | JPEG 编码推流（pop_jpeg） |
| `WebRTCSink` | `src/nodes/sink/webrtc_sink.h` | `WebRTCSinkConfig` | WebRTC 实时推流（libdatachannel + NVENC） |
| `PyNode` | `src/core/py_node.h` | — | Python 自定义节点（callable 包装，自动 GIL 获取） |

## 异常层次

```
VisionPipeError (runtime_error)
├── ConfigError          — 配置参数错误
├── NotFoundError        — Pipeline/节点/模型不存在
├── CudaError            — CUDA 运行时错误
├── ModelLoadError       — 模型加载失败（path, reason）
└── InferError           — 推理执行错误
```

定义：`src/core/error.h`，Python 绑定：`python/bindings/bind_exceptions.cpp`

## Python API 导出

`python/visionpipe/__init__.py` 从 nanobind 扩展 `visionpipe_python` 重导出：

- **枚举**: PipelineState, PipelineStatus, NodeState, OverflowPolicy, DecodeMode
- **数据**: Frame, Detection, Track, Keypoint, PoseResult, QueueStats, NodeStats, PipelineConfig, PipelineStats
- **配置**: SourceConfig, DetectorConfig, ClassifierConfig, YoloSegConfig, RtmPoseConfig, YoloPoseConfig, ByteTrackConfig, AnnotatorConfig, JsonResultSinkConfig, MjpegSinkConfig, WebRTCSinkConfig
- **引擎**: IModelEngine, MockModelEngine, TrtModelEngine
- **节点**: NodeBase, FileSource, RtspSource, DetectorNode, ClassifierNode, YoloSegNode, RtmPoseNode, YoloPoseNode, ByteTrackNode, AnnotatorNode, JsonResultSink, MjpegSink, WebRTCSink
- **管道**: Pipeline, PipelineBuilder, PipelineManager

Python 层额外提供：
- `PyNode`（`python/visionpipe/py_node.py`）— 子类化并覆写 `process(frame)` 实现自定义节点
- `PipelineSpec/NodeSpec/EdgeSpec`（`python/visionpipe/serialization.py`）— pydantic 模型，`export_yaml()`/`load_yaml()` 序列化
- Import 时 monkey-patch：`NodeBase.__rshift__` → PipelineBuilder 链式，`Pipeline.run` → `start()`

## 管理 API (aiohttp)

`python/visionpipe/server/management_api.py` 提供 REST + WebSocket 管理端点：

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/pipelines` | 创建 Pipeline（含 source 时自动启动） |
| GET | `/pipelines` | 列出所有 Pipeline |
| DELETE | `/pipelines/{id}` | 停止并销毁 Pipeline |
| GET | `/pipelines/{id}/health` | 节点级健康状态（processed/errors/fps/queue） |
| POST | `/pipelines/{id}/params` | 设置节点参数（node_id, param_name, value） |
| GET | `/mjpeg/{id}` | MJPEG 视频流（multipart） |
| WS | `/ws/{id}/results` | JSON 推理结果推送 |
| WS | `/ws/{id}/webrtc` | WebRTC SDP/ICE 信令 |
| WS | `/ws/{id}/control` | 控制通道（当前仅 ROI 更新） |

相关文件：`server/schemas.py`（pydantic 请求/响应模型），`server/control_ws.py`，`server/signaling.py`

## 测试结构

```
tests/
├── unit/
│   ├── cpp/          # Google Test
│   │   ├── test_bounded_queue.cpp
│   │   ├── test_logger.cpp
│   │   ├── test_pipeline_dag.cpp
│   │   ├── test_pipeline_manager.cpp
│   │   ├── test_model_registry.cpp
│   │   ├── test_parallel_workers.cpp
│   │   ├── test_bytetrack_node.cpp
│   │   └── test_yolo_seg_node.cpp
│   └── python/       # pytest
│       ├── test_bindings.py
│       ├── test_py_node.py
│       ├── test_yaml_serialization.py
│       └── test_webrtc_sink.py
├── integration/
│   ├── cpp/
│   │   ├── test_trt_engine.cpp
│   │   ├── test_source_nodes.cpp
│   │   ├── test_detector_node.cpp
│   │   ├── test_classifier_node.cpp
│   │   └── test_sinks.cpp
│   └── python/
│       ├── test_management_api.py
│       └── test_roi_hotupdate.py
└── e2e/
    └── python/
        ├── test_multi_pipeline.py
        └── test_webrtc_stream.py
```

测试数据：`tests/data/`（视频文件），`tests/models/`（ONNX/Engine 模型）
下载脚本：`tests/data/download_test_assets.sh`

## CMake 构建目标

| 目标 | 类型 | 来源 |
|------|------|------|
| `visionpipe_core` | 静态库 | `src/core/` |
| `visionpipe_hal` | 库 | `src/hal/`（TRT/CUDA 后端） |
| `visionpipe_nodes` | 库 | `src/nodes/`（所有节点实现） |
| `visionpipe_python` | nanobind 扩展 | `python/bindings/` |

CMake 选项：`VISIONPIPE_BUILD_TESTS=ON`, `BUILD_PYTHON=ON`, `USE_CUDA=ON`, `USE_TENSORRT=ON`, `VISIONPIPE_USE_WEBRTC=OFF`

FetchContent 管理：spdlog 1.14.1, nlohmann_json 3.11.3, googletest 1.14.0, nanobind 2.0.0, libdatachannel 0.20.2（WebRTC 启用时）

## 包管理

- **Python**: uv (pyproject.toml) — `uv pip install -e ".[dev]"`
  - 运行时：pydantic>=2.0, pyyaml>=6.0
  - 开发：pytest, pytest-asyncio, ruff, mypy, httpx, aiohttp, playwright
- **C++ 重依赖**（CUDA、TensorRT、OpenCV with CUDA）：系统包管理器安装
  - OpenCV 需源码编译 `-DWITH_CUDA=ON -DWITH_NVCUVID=ON`（系统包通常不含 CUDA 模块）
- **C++ 轻依赖**：CMake FetchContent 自动下载

## GPU 环境

⚠️ 所有测试均需要真实的 NVIDIA GPU，无 mock GPU 环境。

测试数据位置：`tests/data/`（通过 `tests/data/download_test_assets.sh` 下载）

视频解码策略：Phase 2 使用 `cv::cudacodec::VideoReader`（NVDEC GPU 解码），`DecodeMode::AUTO` 在 NVDEC 不可用时回退 `cv::VideoCapture` CPU 解码。

## 开发规范

详细规范见 `DEV_SPEC.md`，分 6 个阶段（Phase 0-5），规范讨论已全部完成（见 `.claude/spec_review_progress.md`）。

**阶段概览：**

| 阶段 | 目标 | 状态 |
|------|------|------|
| Phase 0 | 工程骨架 + CI 基础 | 大部分完成（T0.2 Frame 字段待改） |
| Phase 1 | C++ 核心调度框架 | 大部分完成（T1.1 节点层次/合并拓扑待重构） |
| Phase 2 | NVIDIA 推理 + 编解码 | 核心功能可用（InferNode batch 接口/Classifier 双模式待实现） |
| Phase 3 | Python 绑定 + DSL | 基础绑定完成（DSL 语义/CustomNode 子进程未实现） |
| Phase 4 | 管理 API + 前端交付 | REST+WS 框架完成（生命周期分离/SinkNode 基类待实现） |
| Phase 5 | 集成验证 + 收尾 | 未开始 |

**阶段门禁（每阶段结束须满足）：**
1. 所有测试通过：`ctest --test-dir build` 和 `uv run pytest`
2. 核心模块覆盖率 >90%，整体 >80%
3. 代码风格检查通过（clang-format + ruff）
4. 无内存泄漏（Valgrind/ASAN）
