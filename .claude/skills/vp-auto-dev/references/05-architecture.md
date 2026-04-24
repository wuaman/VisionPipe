## 5. 系统架构与模块设计

### 5.1 整体架构图

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

#### 各层模块说明

**Python 层**

| 模块 | 说明 |
|---|---|
| Pipeline DSL | 用户面向的编排接口。`pipe = Pipeline()` 创建管道，`src >> det >> biz` 用 `>>` 运算符连接节点构成 DAG，`pipe.run()` 启动执行 |
| Business Nodes | 用户继承 `PyNode` 基类编写的自定义业务逻辑节点（如告警判定、数据落库）。`process(frame)` 方法在 C++ 回调时短暂 acquire GIL 执行 |
| Management API | aiohttp 协程服务，暴露 REST 接口（创建/启动/停止/销毁 pipeline、参数热更、健康检查），是运维和前端的对接入口 |

**nanobind 绑定层**

C++ 对象到 Python 的桥梁。将 Pipeline、Frame、Detection、各 Node 类型暴露为 Python 类，处理 GIL acquire/release，异常从 C++ 穿透到 Python（`VisionPipeError` → `visionpipe.VisionPipeError`）。

**C++ 核心层**

| 模块 | 说明 |
|---|---|
| PipelineManager | 全局管理器，持有所有 pipeline 实例。负责 create/start/stop/destroy 生命周期管理，支持同进程多 pipeline 并行 |
| ModelRegistry | 模型引擎池。按 engine 文件 SHA-256 去重，多 pipeline 共享同一 `IModelEngine` 实例（节省显存）。引用计数 + TTL 过期清理 |
| ControlChannel | 控制通道。WebSocket 接收实时参数（ROI 坐标、阈值），通过 `set_param()` 原子写入节点；REST 路径处理 pipeline CRUD |
| Pipeline (DAG) | 单条 pipeline 的执行引擎。维护节点有向无环图，节点间通过 `BoundedQueue` 连接，异步生产者-消费者模式驱动数据流 |
| SourceNode | 视频源节点（`FileSource` / `RtspSource`）。通过 `DecodeMode` 配置选择 GPU 硬解码（`cv::cudacodec`）或 CPU 软解码（`cv::VideoCapture`） |
| InferNode | 推理节点。持有 `IModelEngine` 的多个 `IExecContext`（`parallel_workers=N`），每个 worker 独立 CUDA stream 并行推理，输出按 frame_id 重排序保证有序 |

**HAL 硬件抽象层**

| 接口 | 说明 | 一期实现 |
|---|---|---|
| IModelEngine | 重资源，代表一个已加载的模型引擎。由 ModelRegistry 管理生命周期，可创建多个推理上下文 | `TrtEngine`（TensorRT） |
| IExecContext | 轻资源，每个 InferNode worker 独占一个。持有独立 CUDA stream，执行 `infer(input, output)` | `TrtExecCtx` |
| IAllocator | 设备内存分配器。线程安全的 `alloc/free`，256 字节对齐 | `CudaAllocator` |
| ICodec（二期） | 编解码 HAL 抽象。`open / decode_next / close`，各平台实现各自的硬件解码器。一期 SourceNode 直接调用 OpenCV，不经过此接口 | 二期实现 |

### 5.2 核心模块说明

#### 5.2.1 节点基类体系

```
BaseNode (C++)
├── SourceNode          # 视频源：RTSP/文件/摄像头
├── InferNode           # 通用推理节点（持有 IExecContext × N workers）
│   ├── DetectorNode    # 目标检测（内置 bbox 后处理）
│   ├── ClassifierNode  # 图像分类（帧内 batch crop）
│   ├── SegmentNode     # 实例分割
│   └── TrackerNode     # 追踪（ByteTrack，纯 CPU）
├── SinkNode            # 输出节点
│   ├── WebRTCSink      # WebRTC 实时预览（核心）
│   ├── JsonResultSink  # 结构化结果推送 WebSocket/HTTP
│   ├── MjpegSink       # MJPEG dev/debug
│   └── RtspSink        # (预留接口，二期)
└── PyNode (nanobind)   # Python 自定义业务节点基类
```

#### 5.2.2 SourceNode 配置

```cpp
enum class DecodeMode {
    AUTO,    // 自动检测：优先 GPU 硬解，不可用时退化为 CPU
    GPU,     // 强制 GPU 硬解（cv::cudacodec / ICodec），不可用时抛异常
    CPU      // 强制 CPU 软解（cv::VideoCapture）
};

struct SourceConfig {
    std::string    uri;            // 文件路径、RTSP URL、设备号
    DecodeMode     decode_mode = DecodeMode::AUTO;
    int            gpu_device   = 0;    // GPU 设备号（多卡场景）
    size_t         queue_capacity = 16; // 输出队列容量
    OverflowPolicy overflow_policy = OverflowPolicy::DROP_OLDEST;
};
```

Python DSL 用法示例：

```python
# GPU 硬解码（默认）
src = FileSource("video.mp4", decode_mode="auto")

# 强制 CPU 软解码
src = FileSource("video.mp4", decode_mode="cpu")

# RTSP GPU 硬解码
src = RtspSource("rtsp://...", decode_mode="gpu")
```

#### 5.2.3 Frame / Tensor 数据结构

##### Frame 累积语义

**Frame 是节点间唯一的数据载体，采用累积（enrichment）模式**：一个 Frame 在整条 pipeline 中只存在一份，每个节点在其上叠加自己的输出，不创建新 Frame，不复制旧 Frame。

```
SourceNode       DetectorNode      ClassifierNode     TrackerNode       PyNode / SinkNode
    │                  │                  │                 │                  │
    ▼                  ▼                  ▼                 ▼                  ▼
frame.image ──────►  读 image        读 image+         读 detections      读全部字段
                     写 detections   读 detections      写 detections      写 user_data
                                     写 detections       (更新 class_id)    （业务逻辑）
                                     (class_id/conf)    写 tracks
```

各节点读写约定：

| 节点类型 | 读取字段 | 写入字段 | 说明 |
|---|---|---|---|
| `SourceNode` | — | `stream_id`, `frame_id`, `pts_us`, `image` | 解码后的原始帧 |
| `DetectorNode` | `image` | `detections[]`（bbox, coarse class_id, confidence） | YOLOv8 检测，class_id 为模型原始类别 |
| `ClassifierNode` | `image`, `detections[]` | `detections[i].class_id`, `detections[i].confidence` | 对每个 detection 的 bbox crop 做细粒度分类，**覆盖** class_id 与 confidence |
| `TrackerNode` | `detections[]` | `tracks[]`, `detections[i].track_id` | ByteTrack 关联，写入轨迹 ID |
| `PyNode` | 任意字段 | `user_data` | Python 业务节点，结果挂载到 user_data |
| `SinkNode` | 任意字段 | — | 只读，不修改 Frame |

**禁止**：节点内不得替换整个 `frame.image`（会泄漏 GPU 内存）；不得拷贝 Frame（编译期已通过 `= delete` 阻止）。

##### Frame 数据结构

```cpp
struct Frame {
    int64_t  stream_id;        // 流标识符，同一 pipeline 内唯一
    int64_t  frame_id;         // 全局单调递增，用于重排序
    int64_t  pts_us;           // 时间戳（微秒）
    Tensor   image;            // GPU / CPU tensor，含 IAllocator 管理的内存
    std::vector<Detection> detections;  // 检测结果，由 DetectorNode 填充
    std::vector<Track>     tracks;      // 追踪结果，由 TrackerNode 填充
    std::any               user_data;  // Python 业务节点附加数据，所有权归 Frame

    // 序列化钩子（预留，用于未来跨进程/跨机传输）
    std::vector<uint8_t> serialize() const;
    static Frame deserialize(const uint8_t* data, size_t len);
};
// 内存所有权：
// - image.data 由 image.allocator 管理，Frame 析构时自动释放
// - user_data 由 std::any 管理，Python 侧持有引用时需保证生命周期
```

Detection 与 Track 结构：

```cpp
struct Detection {
    float   bbox[4];           // [x1, y1, x2, y2]，归一化坐标 [0, 1]
    int     class_id   = 0;    // 类别 ID；ClassifierNode 会覆盖为细粒度分类结果
    float   confidence = 0.f;  // 置信度；ClassifierNode 会覆盖为分类概率
    int64_t track_id   = -1;   // 关联轨迹 ID，-1 表示未追踪
};

struct Track {
    int64_t track_id   = 0;
    int     class_id   = 0;
    float   bbox[4];
    int     age        = 0;    // 轨迹存活帧数
    float   confidence = 0.f;
};
```

##### Tensor 内存类型

```cpp
struct Tensor {
    std::vector<int64_t> shape;          // 如 {1, 3, 640, 640}（NCHW）
    DataType             dtype;          // FLOAT32 / FLOAT16 / INT8 / UINT8
    void*                data;           // 内存指针，由 allocator 管理
    size_t               nbytes;
    IAllocator*          allocator;      // 不拥有，弱引用
};
```

| 内存类型 | `MemoryType` 枚举 | 分配器 | 典型用途 |
|---|---|---|---|
| CUDA 设备显存 | `CUDA_DEVICE` | `CudaAllocator` | 推理输入/输出 tensor，GPU 解码帧 |
| CUDA Pinned 内存 | `CUDA_HOST` | `CudaPinnedAllocator` | H2D / D2H 中转，DMA 加速 |
| CPU 普通内存 | `CPU` | `CpuAllocator`（256 字节对齐） | CPU 软解码帧，Python 侧数据 |

跨节点传递 Tensor 时通过 `std::move` 转移所有权，**不触发内存拷贝**。若下游节点需要 CPU 数据（如 Python 读取），由该节点自行调用 `cudaMemcpy` D2H，不由框架自动转换。

##### Frame 传递机制

```
上游节点 worker_loop                          下游节点 worker_loop
─────────────────────────                    ─────────────────────────
process(frame)                               pop_for(100ms) → frame_opt
  │                                            │
  ▼                                            ▼
output_queue_->push(std::move(frame))  ←──  input_queue_ (同一个对象)
```

- Frame 通过 `std::move` 入队，**零拷贝**，大 tensor 不会被复制
- `pop_for()` 返回 `std::optional<Frame>`，超时或队列停止时返回 `nullopt`
- 节点间队列容量默认 16，溢出策略默认 `DROP_OLDEST`（实时流保低延迟）

#### 5.2.4 BoundedQueue

```cpp
template<typename T>
class BoundedQueue {
public:
    enum class OverflowPolicy { DROP_OLDEST, DROP_NEWEST, BLOCK };

    // capacity: 队列最大容量，必须 > 0
    // policy: 溢出策略
    BoundedQueue(size_t capacity, OverflowPolicy policy);

    // 线程安全：可由多生产者/多消费者并发调用
    // DROP_OLDEST: 队列满时丢弃最老元素，push 永不阻塞
    // DROP_NEWEST: 队列满时丢弃新元素，push 永不阻塞
    // BLOCK: 队列满时阻塞直到有空间
    void push(T item);

    // 非阻塞弹出，队列为空时返回 nullopt
    std::optional<T> pop();

    // 阻塞弹出，队列空时阻塞直到有数据
    T pop_blocking();

    QueueStats stats() const;   // 占用率、丢帧计数、FPS
};
```

#### 5.2.5 ModelRegistry

```cpp
class ModelRegistry {
public:
    static ModelRegistry& instance();

    // 获取模型引擎，按 engine_path 内容 SHA-256 去重
    // 首次 acquire 时加载，后续 acquire 增加引用计数
    // 抛出 ModelLoadError 如果文件不存在或格式错误
    std::shared_ptr<IModelEngine> acquire(const std::string& engine_path);

    // 释放引用，ref_count 减 1
    // ref_count=0 后，引擎在 TTL 期间保留，超期后由 gc_loop 清理
    void release(const std::string& key);

    // 设置空闲模型 TTL（默认 60 秒）
    void set_ttl(std::chrono::seconds ttl);

private:
    std::string sha256_file(const std::string& path);
    void gc_loop();             // 后台线程，清理 TTL 过期的空闲模型

    struct Entry {
        std::shared_ptr<IModelEngine> engine;
        std::atomic<int>              ref_count{0};
        std::chrono::steady_clock::time_point last_release;
    };
    std::unordered_map<std::string, Entry> pool_;
    std::mutex mu_;
    std::chrono::seconds ttl_{60};  // 可配置
};

// 自定义异常类
class ModelLoadError : public std::runtime_error {
public:
    explicit ModelLoadError(const std::string& path, const std::string& reason);
};
```

#### 5.2.6 PipelineManager

```cpp
class PipelineManager {
public:
    // 创建 pipeline，返回唯一 ID
    // 抛出 ConfigError 如果配置非法
    std::string create(const PipelineConfig& cfg);

    // 启动 pipeline，id 不存在时抛出 NotFoundError
    void start(const std::string& id);

    // 停止 pipeline（触发 DRAINING），id 不存在时抛出 NotFoundError
    void stop(const std::string& id);

    // 销毁 pipeline（必须先 stop）
    void destroy(const std::string& id);

    // 查询状态
    PipelineStatus status(const std::string& id) const;

    // 列出所有 pipeline
    std::vector<PipelineInfo> list() const;
};

enum class PipelineStatus { INIT, RUNNING, DRAINING, STOPPED, ERROR };
```

#### 5.2.7 HAL 接口

```cpp
// 重资源：ModelRegistry 管理，跨 pipeline 共享
// 生命周期：由 ModelRegistry 管理，用户通过 acquire/release 获取/释放
class IModelEngine {
public:
    // 创建推理上下文，每个 worker 独立持有
    // 返回的 IExecContext 生命周期由调用者管理
    // 抛出 CudaError 如果 GPU 资源不足
    virtual std::unique_ptr<IExecContext> create_context() = 0;

    // 返回模型占用的 GPU 显存字节数
    virtual size_t device_memory_bytes() const = 0;

    virtual ~IModelEngine() = default;
};

// 轻资源：每个 InferNode worker 独占
// 线程安全：每个 worker 独占一个 context，无需加锁
class IExecContext {
public:
    // 执行推理
    // input: 输入张量，shape 和 dtype 必须匹配模型输入
    //   - 默认：[1, 3, H, W]，NCHW，float32，CUDA device memory
    // output: 输出张量，由 infer 填充
    //   - 默认：[1, num_outputs, H*W]，float32，CUDA device memory
    // 抛出 InferError 如果输入 shape 不匹配或推理失败
    // 抛出 CudaError 如果 CUDA 操作失败
    virtual void infer(const Tensor& input, Tensor& output) = 0;

    virtual ~IExecContext() = default;
};

// 设备内存分配器
// 线程安全：实现类必须保证 alloc/free 可并发调用
class IAllocator {
public:
    // 分配内存
    // bytes: 分配字节数，必须 > 0
    // 返回: 内存指针，失败返回 nullptr（不抛异常）
    // 对齐：默认 256 字节（适配 CUDA 合作组）
    virtual void* alloc(size_t bytes) = 0;

    // 释放内存，ptr 为 nullptr 时无操作
    virtual void free(void* ptr) = 0;

    // 返回内存类型
    virtual MemoryType type() const = 0;  // CUDA_DEVICE / CUDA_HOST / ACL / RKNN_DMA

    virtual ~IAllocator() = default;
};

// 自定义异常类
class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t err);
    cudaError_t error_code() const;
};

class InferError : public std::runtime_error {
public:
    explicit InferError(const std::string& reason);
};
```

#### 5.2.8 ControlChannel（管理 API + 参数热更）

- 内嵌 aiohttp 协程服务，与 C++ Pipeline 同进程运行，监听独立端口（默认 8080）
- REST 接口：POST /pipelines、GET /pipelines、DELETE /pipelines/{id}、POST /pipelines/{id}/params、GET /pipelines/{id}/health
- WebSocket 接口 /ws/{pipeline_id}/control：接收 ROI 坐标、阈值等实时参数
- C++ 层 NodeBase::set_param(name, value) 使用 std::atomic 或 double-buffer，下一帧原子生效

---
