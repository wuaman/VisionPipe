# VisionPipe-py API 参考

> 适用版本：`visionpipe.__version__ = 0.1.0`（Phase 5 收尾）
>
> 本文档涵盖 Python 层公开 API。底层 C++/CUDA/TensorRT 细节请参考 `DEV_SPEC.md`。

## 目录

- [1. 概览](#1-概览)
- [2. 数据结构](#2-数据结构)
- [3. 枚举类型](#3-枚举类型)
- [4. 节点目录](#4-节点目录)
  - [4.1 Source 节点](#41-source-节点)
  - [4.2 Infer 节点](#42-infer-节点)
  - [4.3 Tracker 节点](#43-tracker-节点)
  - [4.4 Visualize 节点](#44-visualize-节点)
  - [4.5 Sink 节点](#45-sink-节点)
  - [4.6 自定义节点](#46-自定义节点)
- [5. Pipeline 与生命周期](#5-pipeline-与生命周期)
- [6. DSL 速查](#6-dsl-速查)
- [7. YAML 序列化](#7-yaml-序列化)
- [8. 管理 API](#8-管理-api)
- [9. 异常层次](#9-异常层次)
- [10. 端到端示例](#10-端到端示例)

---

## 1. 概览

`visionpipe` 包对外暴露的核心符号一览（见 `python/visionpipe/__init__.py`）：

| 分类 | 符号 |
|------|------|
| 枚举 | `PipelineState`, `PipelineStatus`, `NodeState`, `OverflowPolicy`, `DecodeMode` |
| 数据 | `Frame`, `Detection`, `Classification`, `Track`, `QueueStats`, `NodeStats`, `PipelineStats`, `PipelineConfig` |
| 配置 | `SourceConfig`, `DetectorConfig`, `ClassifierConfig`, `SegmentConfig`, `ByteTrackConfig`, `AnnotatorConfig`, `JsonResultSinkConfig`, `MjpegSinkConfig`, `WebRTCSinkConfig` |
| HAL  | `IModelEngine`, `MockModelEngine`, `TrtModelEngine` |
| 节点基类 | `NodeBase`, `SourceNode`, `SinkNode`, `ProcessProxyNode`, `PyNode`, `CustomNode`, `FrameView` |
| 节点实现 | `FileSource`, `RtspSource`, `DetectorNode`, `ClassifierNode`, `SegmentNode`, `ByteTrackNode`, `AnnotatorNode`, `JsonResultSink`, `MjpegSink`, `WebRTCSink` |
| 编排 | `Pipeline`, `PipelineBuilder`, `PipelineManager` |
| 序列化 | `PipelineSpec`, `NodeSpec`, `EdgeSpec`, `Pipeline.export_yaml`, `Pipeline.load_yaml`, `Pipeline.from_yaml`, `Pipeline.rebuild_from_spec` |
| 异常 | `VisionPipeError`, `ConfigError`, `NotFoundError`, `CudaError`, `ModelLoadError`, `InferError`, `StreamError` |

---

## 2. 数据结构

### `Frame`

每帧在 DAG 中流动的载体，所有节点直接读写其字段。Frame 是 **move-only**（不可在 Python 层重复持有引用）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `stream_id` | `str` | 来源标识（多 Source 合并时区分流） |
| `frame_id` | `int` | Source 端单调递增的帧编号 |
| `pts_us` | `int` | 帧时间戳（微秒） |
| `image` | `Tensor`（C++ 持有） | GPU 或 CPU 上的图像；通过 `frame.image_numpy()` 在 CPU 形态时读取为 numpy |
| `detections` | `list[Detection]` | DetectorNode 写入 |
| `classifications` | `list[Classification]` | ClassifierNode 写入（与 `detections` 通过 `detection_index` 关联） |
| `tracks` | `list[Track]` | ByteTrackNode 写入 |
| `user_data` | `dict[str, Any]` | 各 PyNode/CustomNode 用独立 key 存放业务数据，互不覆盖 |

辅助方法（C++ 暴露给 Python）：

- `frame.has_image() -> bool`
- `frame.image_numpy() -> np.ndarray`（仅当 image 已在 CPU）
- `frame.has_user_data(key) -> bool`
- `frame.get_user_data(key) -> Any`
- `frame.set_user_data(key, value) -> None`

### `Detection`

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox` | `list[float]` | `[x1, y1, x2, y2]`（像素坐标） |
| `class_id` | `int` | 模型类别索引 |
| `confidence` | `float` | 置信度 ∈ [0, 1] |
| `track_id` | `int` | 由 Tracker 回填；DetectorNode 输出时为 0/未定义 |

### `Classification`

| 字段 | 类型 | 说明 |
|------|------|------|
| `detection_index` | `int` | 关联的 `frame.detections[i]` 下标；整图分类时为 `-1` |
| `class_id` | `int` | 分类结果类别 |
| `confidence` | `float` | 概率 |

### `Track`

| 字段 | 类型 | 说明 |
|------|------|------|
| `track_id` | `int` | 跨帧持续的轨迹 ID（> 0） |
| `class_id` | `int` | 沿用关联 detection 的类别 |
| `bbox` | `list[float]` | 与 detection 同坐标系 |
| `age` | `int` | 该轨迹被持续匹配的帧数 |
| `confidence` | `float` | 关联 detection 的置信度 |

### 统计结构

| 名称 | 关键字段 |
|------|----------|
| `QueueStats` | `capacity`, `size`, `pushed`, `popped`, `dropped`, `overflow_policy` |
| `NodeStats` | `name`, `state`, `frames_processed`, `errors`, `fps`, `latency_ms` |
| `PipelineStats` | `name`, `state`, `node_stats: list[NodeStats]` |

---

## 3. 枚举类型

| 枚举 | 取值 |
|------|------|
| `PipelineState` / `PipelineStatus` | `INIT`, `RUNNING`, `DRAINING`, `STOPPED`, `ERROR` |
| `NodeState` | `INIT`, `RUNNING`, `STOPPED`, `ERROR` |
| `OverflowPolicy` | `BLOCK`, `DROP_OLDEST`, `DROP_NEWEST` |
| `DecodeMode` | `AUTO`, `GPU`, `CPU` |

---

## 4. 节点目录

### 4.1 Source 节点

#### `SourceConfig`

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `uri` | `str` | — | 文件路径或 RTSP URL |
| `decode_mode` | `DecodeMode` | `AUTO` | 解码后端 |
| `gpu_device` | `int` | `0` | CUDA 设备号 |
| `queue_capacity` | `int` | `8` | 下游 input_queue 容量 |
| `overflow_policy` | `OverflowPolicy` | `DROP_OLDEST` | 队列溢出策略 |
| `loop` | `bool` | `False` | 文件源播放结束后循环 |
| `skip_frames` | `int` | `0` | 解码后每 N 帧丢弃 N-1 帧（降采样） |
| `max_retries` | `int` | `0` | RTSP 断流重连次数 |
| `retry_interval_ms` | `int` | `1000` | 重连间隔（毫秒） |
| `stream_id` | `str` | `""` | 合并拓扑下区分来源 |

#### `FileSource(config: SourceConfig)`

本地视频源，使用 `cv::cudacodec`（GPU 路径）或 `cv::VideoCapture`（CPU 路径）。`DecodeMode.AUTO` 时自动选择，`GPU` 强制硬解（不可用时抛 `CudaError`）。

#### `RtspSource(config: SourceConfig)`

RTSP 流源，CPU 路径解码后通过 `GpuMat::upload()` 提升到 GPU。支持断流自动重连。

### 4.2 Infer 节点

所有 Infer 节点继承自 C++ 的 `InferNode`，内部使用 `parallel_workers` 并行执行，输出按 `frame_id` 重排序。

#### `DetectorConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `input_height` | `640` | 模型输入高 |
| `input_width` | `640` | 模型输入宽 |
| `score_threshold` | `0.25` | 分数阈值 |
| `nms_threshold` | `0.45` | NMS IoU 阈值 |
| `max_detections` | `300` | 单帧最大检测目标数 |
| `workers` | `1` | 并行 worker 数（每 worker 独立 `IExecContext`） |

`DetectorNode(engine: IModelEngine, config: DetectorConfig, name: str = "detector")`

运行时热更新：

```python
detector.set_param("score_threshold", 0.5)
detector.set_param("roi", [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
detector.set_param("clear_roi", True)
```

#### `ClassifierConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `input_height` / `input_width` | `224` | 模型输入尺寸 |
| `target_classes` | `[]` | **非空**=二级分类（按 detections 裁切）；**空**=整图分类（`detection_index = -1`） |
| `max_batch_size` | `16` | 单次推理最大 batch（动态攒帧） |
| `normalize_mean_std` | `True` | ImageNet 均值方差归一化 |
| `workers` | `1` | 并行 worker 数 |

`ClassifierNode(engine: IModelEngine, config: ClassifierConfig, name: str = "classifier")`

#### `SegmentConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `input_height` / `input_width` | `640` | 模型输入尺寸 |
| `score_threshold` | `0.25` | 分数阈值 |
| `nms_threshold` | `0.45` | NMS IoU |
| `mask_threshold` | `0.5` | 掩码二值化阈值 |
| `max_detections` | `100` | 最大目标数 |
| `workers` | `1` | 并行 worker 数 |

`SegmentNode(engine: IModelEngine, config: SegmentConfig, name: str = "segment")`

输出同时填充 `frame.detections`（bbox/类别）和实例掩码（C++ 内部的 `Frame::masks`）。

### 4.3 Tracker 节点

#### `ByteTrackConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `track_thresh` | `0.5` | 高分检测阈值 |
| `match_thresh` | `0.8` | 关联 IoU 阈值 |
| `track_buffer` | `30` | 丢失轨迹保留帧数 |
| `frame_rate` | `30` | 视频帧率 |

`ByteTrackNode(config: ByteTrackConfig, name: str = "bytetrack")`

纯 CPU 实现，依赖上游 `frame.detections`，回填 `frame.tracks` 与 `Detection.track_id`。

### 4.4 Visualize 节点

#### `AnnotatorConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `draw_detections` | `True` | 绘制 bbox + 类别名 + 分数 |
| `draw_tracks` | `True` | 绘制 track_id |
| `draw_masks` | `False` | 叠加实例分割掩码 |
| `mask_alpha` | `0.5` | 掩码透明度 |
| `class_names` | `{}` | `dict[int, str]`，未指定时用 `class_<id>` 占位 |

`AnnotatorNode(config: AnnotatorConfig, name: str = "annotator")`

在 `frame.image`（CPU 形态）上原地绘制。GPU 形态图像会跳过绘制（不报错）。

### 4.5 Sink 节点

所有 Sink 继承自 `SinkNode`，统一暴露 `enabled` 属性。运行时通过 `sink.set_param("enabled", True/False)` 切换；`enabled=False` 时跳过实际工作，不消耗 CPU/GPU 资源。

#### `JsonResultSinkConfig` / `JsonResultSink`

| 字段 | 默认 | 说明 |
|------|------|------|
| `include_detections` | `True` | JSON 含 `detections` 字段 |
| `include_tracks` | `True` | JSON 含 `tracks` 字段 |
| `buffer_capacity` | `1024` | 内部 BoundedQueue 容量 |

```python
sink = JsonResultSink(JsonResultSinkConfig(), "sink")
# 阻塞最多 200ms，无数据时返回 None
payload: str | None = sink.pop_json(timeout_ms=200)
```

`enabled` 默认 `True`。

#### `MjpegSinkConfig` / `MjpegSink`

| 字段 | 默认 | 说明 |
|------|------|------|
| `jpeg_quality` | `80` | JPEG 编码质量（0-100） |
| `buffer_capacity` | `4` | 最新帧环形队列容量 |

```python
sink = MjpegSink(MjpegSinkConfig(), "mjpeg")
jpeg_bytes: bytes | None = sink.pop_jpeg(timeout_ms=200)
```

`enabled` 默认 `False`（按规范作为调试/降级 Sink），需通过 `sink.set_param("enabled", True)` 或 REST `POST /pipelines/{id}/params` 显式开启。

#### `WebRTCSinkConfig` / `WebRTCSink`

| 字段 | 默认 | 说明 |
|------|------|------|
| `use_nvenc` | `True` | 启用 NVENC 硬编 H.264 |
| `video_bitrate_kbps` | `2000` | 目标码率 |
| `fps` | `30` | 编码帧率 |
| `keyframe_interval` | `60` | 关键帧间隔（帧） |
| `stun_server` | `"stun:stun.l.google.com:19302"` | STUN 服务器 |

需在 cmake 时启用 `-DVISIONPIPE_USE_WEBRTC=ON`。Python 侧 SDP/ICE 信令通过 `ManagementServer` 的 `/ws/{id}/webrtc` 端点。

### 4.6 自定义节点

#### `PyNode`（同进程回调）

`from visionpipe import PyNode`

```python
class CountNode(PyNode):
    def __init__(self) -> None:
        super().__init__(name="count")

    def process(self, frame) -> None:
        frame.set_user_data("n_dets", len(frame.detections))
```

- C++ 调用时自动 `gil_scoped_acquire`
- 通过 `node._cpp_node` 在 `Pipeline.add_node()` 和 `connect()` 中使用
- 支持 `>>` 链式语法（`PyNode.__rshift__` 自动 unwrap 到 `_cpp_node`）

#### `CustomNode`（子进程，推荐用于重逻辑）

`from visionpipe import CustomNode, FrameView`

```python
class HeavyAnalysisNode(CustomNode):
    def on_frame(self, frame: FrameView) -> None:
        # 跑独立进程，无 GIL 限制；可调用任意 Python 库
        frame.user_data["analysis"] = my_heavy_compute(frame.detections)

# 用法
node = HeavyAnalysisNode(name="analysis", process_mode="subprocess")
pipeline.add_node(node._cpp_node)
```

参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `name` | `"custom_node"` | 节点名 |
| `process_mode` | `"subprocess"` | `"subprocess"` 或 `"inline"` |
| `restart_limit` | `3` | 子进程崩溃自动重启次数 |
| `restart_delay` | `1.0` | 重启延迟（秒） |

`FrameView` 是子进程端的安全视图：

- 只读字段：`frame_id`, `stream_id`, `pts_us`, `detections`, `tracks`
- 可写字段：`user_data: dict`（赋值后由 IPC 同步回主进程）
- `on_frame` 返回后视图自动失效，禁止在外部线程持有

退出时务必调用 `node.stop()` 释放子进程。

---

## 5. Pipeline 与生命周期

### `PipelineConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `name` | `""` | Pipeline 名（用于日志/管理 API） |
| `default_queue_capacity` | `8` | 自动建队列时使用的容量 |
| `default_overflow_policy` | `DROP_OLDEST` | 自动建队列的溢出策略 |

### `Pipeline`

```python
pipeline = Pipeline(PipelineConfig())
pipeline.add_node(node)              # 注册节点（自动建立 input_queue / output_queue）
pipeline.connect(upstream, downstream)
# 多 Source 合并：多次 connect(src_i, downstream) 共享 downstream.input_queue
pipeline.validate_dag()              # 检测环路；抛 ConfigError
pipeline.start()                     # 启动所有节点 worker
pipeline.run(block=False)            # = start()；block=True 则等待 source 自然结束
pipeline.wait_stop()                 # 阻塞至 STOPPED
pipeline.stop(drain=True)            # DRAINING → STOPPED
pipeline.stats() -> PipelineStats
pipeline.state() -> PipelineState
pipeline.nodes() -> dict[str, NodeBase]
```

### `PipelineManager`

```python
manager = PipelineManager()
pid: str = manager.create_pipeline(pipeline)  # 接管 pipeline 的生命周期
manager.start(pid)
status = manager.status(pid)                  # PipelineStatus 枚举
manager.stop(pid)
manager.destroy(pid)                          # 必须先 STOPPED
manager.list() -> list[str]
manager.get(pid) -> Pipeline
```

`PipelineManager` 内置 `ModelRegistry`：多条 pipeline 通过同一 `IModelEngine` 实例参与时自动复用模型权重，节省显存。

---

## 6. DSL 速查

`>>` 运算符在 `python/visionpipe/__init__.py` 中以 monkey-patch 实现：

```python
# 线性链
pipe = source >> detector >> sink

# 链式追加（Pipeline 端）
pipe = source >> detector
pipe = pipe >> tracker >> sink

# 合并拓扑（多 source → 一个下游）
pipe = [src1, src2, src3] >> detector >> sink

# PyNode / CustomNode 自动 unwrap _cpp_node
pipe = source >> detector >> MyPyNode("count") >> sink
```

`>>` 直接返回 `Pipeline`（不需要 `.build()`）。Pipeline 内部维护 `_tail` 指针支持继续链式追加。

启动入口（同义）：

```python
pipe.run(block=False)         # 非阻塞，立即返回
pipe.run(block=True)          # 阻塞至 source 自然结束（适合文件视频）
pipe.start(); pipe.wait_stop()
pipe.stop(drain=True)
```

---

## 7. YAML 序列化

```python
# 导出当前 pipeline 拓扑 + 节点参数到 YAML
pipe.export_yaml("pipeline.yaml")

# 仅解析 YAML 得到 PipelineSpec（不构建 Pipeline，不需要 GPU）
spec: PipelineSpec = Pipeline.load_yaml("pipeline.yaml")
for n in spec.nodes:
    print(n.name, n.type, n.params)
for e in spec.edges:
    print(e.from_node, "->", e.to_node)

# 完整重建：node_overrides 用于注入需要外部依赖的节点（如 DetectorNode 需要 engine）
rebuilt = Pipeline.from_yaml(
    "pipeline.yaml",
    node_overrides={
        "src": FileSource(SourceConfig("...")),
        "det": DetectorNode(engine, DetectorConfig(), "det"),
        "sink": JsonResultSink(JsonResultSinkConfig(), "sink"),
    },
)
```

`NodeSpec` 字段：

| 字段 | 说明 |
|------|------|
| `name` | 节点名 |
| `type` | 内置类型字符串（`file_source` / `detector` / `json_result_sink` 等） |
| `params` | 节点配置参数 dict |
| `module` | CustomNode 专用，待自动 import 的 Python 模块 |
| `class_name` | CustomNode 专用，类名 |
| `process_mode` | CustomNode 专用，`subprocess` / `inline` |

YAML 解析器（pydantic）会拦截非法节点类型并抛 `ConfigError`。

---

## 8. 管理 API

`from visionpipe.server.management_api import ManagementServer`

```python
manager = visionpipe.PipelineManager()
server = ManagementServer(manager, host="0.0.0.0", port=8080)
await server.start()
# ... 业务 ...
await server.stop()
```

### REST 端点

| 方法 | 路径 | 状态码 | 说明 |
|------|------|--------|------|
| `POST` | `/pipelines` | 201 | body `{"spec": <PipelineSpec dict or YAML str>}`，返回 `{"id": "..."}` |
| `GET` | `/pipelines` | 200 | `[{id, name, state}, ...]` |
| `POST` | `/pipelines/{id}/start` | 200 | 转 INIT/STOPPED → RUNNING |
| `POST` | `/pipelines/{id}/stop` | 200 | RUNNING → DRAINING → STOPPED |
| `DELETE` | `/pipelines/{id}` | 204 | 必须先 STOPPED，否则 409 |
| `GET` | `/pipelines/{id}/health` | 200 | `{nodes: [{name, queue: QueueStats, fps}, ...]}` |
| `GET` | `/pipelines/{id}/nodes` | 200 | `[NodeStats, ...]`（含 `fps`/`latency_ms`/`state`/`frames_processed`/`errors`） |
| `POST` | `/pipelines/{id}/params` | 200 | body `{node_id, param_name, value}` → 转发到 `NodeBase.set_param()` |
| `GET` | `/mjpeg/{id}` | 200 | multipart MJPEG 流（依赖 MjpegSink 且 `enabled=True`） |

### WebSocket 端点

| 路径 | 协议 | 说明 |
|------|------|------|
| `/ws/{id}/results` | 文本 JSON | 推送 JsonResultSink 每帧结果 |
| `/ws/{id}/control` | 文本 JSON | 通用控制通道（见下） |
| `/ws/{id}/webrtc` | 文本 JSON | SDP/ICE 信令（仅在 WebRTCSink 启用时可用） |

#### `/ws/{id}/control` 消息格式

请求：

```json
{"type": "ping"}
{"type": "set_param", "node_id": "det", "param_name": "score_threshold", "value": 0.5}
{"type": "roi", "node_id": "det", "polygons": [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]], "coord": "normalized"}
```

响应：

```json
{"type": "pong"}
{"type": "ack", "ref_type": "set_param"}
{"type": "error", "message": "node not found: no_such_node"}
```

`set_param` 内部直接调用对应节点的 `set_param(param_name, value)`，参数类型由节点自己解释（数字、字符串、列表等）。

---

## 9. 异常层次

```
VisionPipeError (RuntimeError 子类)
├── ConfigError          配置或参数错误
├── NotFoundError        Pipeline/Node/模型不存在
├── CudaError            CUDA / NVDEC 等 GPU 运行时错误
├── ModelLoadError       模型加载失败 (file, reason)
├── InferError           推理执行错误
└── StreamError          视频源连接 / 断流错误
```

所有异常通过 nanobind 自动穿透到 Python；可用 `try/except visionpipe.ConfigError:` 精确捕获。

---

## 10. 端到端示例

### 10.1 单 Pipeline 检测 + JSON 输出

```python
import json, time, visionpipe as vp

src = vp.FileSource(vp.SourceConfig("video.mp4"))
engine = vp.TrtModelEngine("models/yolov8n_dynamic.engine")
det = vp.DetectorNode(engine, vp.DetectorConfig(), "det")
sink = vp.JsonResultSink(vp.JsonResultSinkConfig(), "sink")

pipe = src >> det >> sink
pipe.run(block=False)

deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    payload = sink.pop_json(200)
    if payload:
        print(json.loads(payload)["frame_id"])
pipe.stop()
```

### 10.2 多 Pipeline 共享 Engine

参见 `examples/multi_pipeline_demo.py` —— 两条 Pipeline 共享同一 `TrtModelEngine`，分别 filter 出 vehicle / person 类别，验证 ModelRegistry 复用 + 生命周期隔离。

### 10.3 REST 管理 + WebSocket 控制

```python
import asyncio, aiohttp, visionpipe as vp
from visionpipe.server.management_api import ManagementServer

async def main() -> None:
    mgr = vp.PipelineManager()
    server = ManagementServer(mgr, host="127.0.0.1", port=8080)
    await server.start()
    async with aiohttp.ClientSession() as sess:
        await sess.post("http://127.0.0.1:8080/pipelines", json={"spec": {...}})
        # ... start / monitor / set_param via /ws/{id}/control ...
    await server.stop()

asyncio.run(main())
```

完整的三层 E2E 验证脚本见 `tests/e2e/test_e2e_validation.py`，可直接作为更复杂用法的范本。

---

## 附录：常见问题

- **`OverflowPolicy` 选择**：实时流默认 `DROP_OLDEST` 保低延迟；离线文件 + `InferNode`（结果按 frame_id 重排）必须用 `BLOCK`，否则 frame_id 空洞会卡死下游重排序队列。
- **GPU 解码失败**：检查 OpenCV 是否带 `WITH_CUDA=ON WITH_NVCUVID=ON` 编译；`DecodeMode.AUTO` 会自动回退 CPU 路径。
- **CustomNode 子进程没退出**：脚本结束前确保调用 `node.stop()`；否则 IPC socket 不释放可能残留僵尸进程。
- **`PipelineManager.destroy` 报 409**：先 `stop()` 转到 STOPPED 状态再 destroy；REST 接口同样要求。
- **`Pipeline.run(block=True)` 一直不返回**：检查 source 是否设置了 `loop=True` 或是 RTSP 这类无尽源；非阻塞模式请用 `run(block=False)` + 自己控制停止时机。
