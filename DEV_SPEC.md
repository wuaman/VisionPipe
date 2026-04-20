# VisionPipe-py DEV_SPEC

> 版本：v0.1-draft
> 日期：2026-04-20
> 状态：讨论确认中

---

## 1. 项目概述

### 1.1 项目定位

VisionPipe-py 是一个面向生产环境的**视频 AI 推理框架**，底层由 C++ 驱动以保证高性能，业务层由 Python 实现以保证灵活性。框架以**有向无环图（DAG）节点管道**为核心抽象，用户通过 Python DSL 编排节点，框架负责调度、并发、资源管理和硬件适配。

一期聚焦 NVIDIA GPU（TensorRT + NVDEC/NVENC），架构预留标准硬件抽象层（HAL），后续可平滑扩展至华为昇腾、瑞芯微 RKNN 等异构推理卡。

### 1.2 设计理念

- **热路径全在 C++**：编解码、预处理、模型推理、后处理、追踪全部运行在 C++ 线程池中，与 Python GIL 无关。
- **Python 只做"最后一公里"**：Pipeline 编排、自定义业务节点、告警规则、外部系统对接均在 Python 层实现，开发体验友好。
- **节点图 = 生产者-消费者链**：每个节点持有有界输入队列，节点间异步解耦，瓶颈节点可配置 `parallel_workers` 横向扩展。
- **同进程多 Pipeline**：多个业务场景（如物理实验、化学实验）在同一进程内并行运行，共享模型权重，优雅启停，无需 Docker 隔离。
- **硬件无关的模型管理**：ModelRegistry 以引擎文件内容 SHA-256 为键做模型去重和引用计数，HAL 接口屏蔽厂商差异。

---

## 2. 核心特点

| # | 特点 | 说明 |
|---|---|---|
| 1 | **Python DSL 编排** | 用 `>>` 运算符连接节点构图，可导出/导入 YAML 用于版本化和运维下发 |
| 2 | **C++ 热路径，零 GIL 干扰** | 推理、编解码、调度全在 C++ 线程池；Python 业务节点回调时短暂 acquire GIL |
| 3 | **同进程多 Pipeline** | PipelineManager 支持动态创建/销毁多条 pipeline，内嵌 REST 管理 API，无需 Docker 隔离 |
| 4 | **模型去重复用** | ModelRegistry 按引擎文件 SHA-256 去重，多条 pipeline 共享同一 `IModelEngine`，VRAM 不重复占用 |
| 5 | **优雅启停协议** | DRAINING → teardown → STOPPED 三段式退出，GPU 资源安全释放，典型耗时 <500ms |
| 6 | **节点并发扩展** | 瓶颈节点配置 `parallel_workers=N`，多个 worker 共享模型权重独立执行上下文，真 GPU 并行 |
| 7 | **有界队列 + 溢出策略** | 每节点有界输入队列；实时流默认 `DROP_OLDEST` 保低延迟，文件处理可选 `BLOCK` 不丢帧 |
| 8 | **ROI 实时热更** | 前端 canvas 框选 → WebSocket 归一化坐标 → C++ `set_param()` 原子写 → 下一帧生效 |
| 9 | **HAL 硬件抽象** | `IModelEngine` / `IExecContext` / `IAllocator` 三接口屏蔽厂商；扩展新硬件只需实现三个类 |
| 10 | **内置可观测性** | 每节点暴露队列占用率、丢帧计数、FPS；健康接口 `GET /pipelines/{id}/health`；spdlog 结构化日志 |

---

## 3. 技术选型

### 3.1 核心依赖

| 层次 | 组件 | 版本要求 | 用途 |
|---|---|---|---|
| 推理 | TensorRT | >=8.6 | 模型编译与 GPU 推理 |
| 编解码（GPU） | NVDEC / NVENC（via FFmpeg with CUVID） | CUDA >=11.8 | 硬件视频编解码 |
| 编解码（CPU） | FFmpeg | >=5.0 | CPU 软解码 / 容器封装 |
| 图像处理 | OpenCV | >=4.7（CUDA build） | 预处理、可视化 |
| GPU 计算 | CUDA | >=11.8 | CUDA stream 管理、内存分配 |
| 追踪算法 | ByteTrack（内置 C++ 实现） | — | 多目标追踪 |
| Python 绑定 | nanobind | >=1.9 | C++ ↔ Python 互调，含 `gil_scoped_release` |

### 3.2 构建与工程

| 维度 | 选型 | 说明 |
|---|---|---|
| C++ 标准 | C++17 | `optional`/`variant`/`filesystem`/结构化绑定 |
| 构建系统 | CMake 3.20+ | CUDA 原生支持，FetchContent 管理轻依赖 |
| 包管理 | 系统包 + FetchContent | 重依赖系统安装；spdlog/nlohmann-json/cpp-httplib 由 FetchContent 拉取 |
| Python 版本 | >=3.10 | match 语法，类型注解完善 |
| C++ 日志 | spdlog | 异步模式，支持 JSON 结构化输出 |
| Python 日志 | logging + structlog | 与 spdlog 统一输出格式 |
| 序列化 | nlohmann/json (C++) + pydantic v2 (Python) | 管理 API 数据格式、Pipeline YAML 导出 |
| WebRTC | libdatachannel | 轻量 C++ 原生，无 Chromium 依赖 |
| 管理 API | aiohttp（协程，同进程） | 暴露 Pipeline CRUD 和健康检查接口 |
| 测试（C++） | Google Test + Google Mock | 单元测试 / 集成测试 |
| 测试（Python） | pytest + pytest-asyncio | Python 层及 E2E 测试 |
| CI | GitHub Actions | Ubuntu 22.04，CUDA mock 环境单元测试，实机集成测试 |
| 目标 OS | Linux（Ubuntu 22.04/24.04） | 服务器端及 Jetson 均为 Linux |
| License | Apache 2.0 | 依赖均兼容（避免 GPL FFmpeg 选项） |

### 3.3 一期验证模型

| 任务 | 模型 | 优先级 | 交付物 |
|---|---|---|---|
| 目标检测 | YOLOv8 / YOLOv11 | P0 | ONNX→TRT 转换脚本 + 后处理插件 + benchmark |
| 图像分类 | ResNet50 / EfficientNet-B0 | P0 | 同上 |
| 实例分割 | YOLOv8-seg | P1 | 同上 |
| 目标追踪 | ByteTrack | P1 | 内置 C++ 实现，无需 GPU |
| 语义分割 | — | 二期 | — |
| 姿态估计 | — | 二期 | — |

---

## 4. 测试方案

### 4.1 测试理念

采用 **TDD（测试驱动开发）** 理念：先写测试用例定义接口契约，再实现功能。所有新功能 PR 必须附带对应测试，CI 强制 gate。

### 4.2 分层测试结构

tests/
├── unit/           # 单元测试：纯逻辑，无 GPU 依赖，可在 CI 无卡环境运行
│   ├── cpp/        # Google Test
│   └── python/     # pytest
├── integration/    # 集成测试：需真实 GPU，测节点间交互和 Pipeline 生命周期
│   ├── cpp/
│   └── python/
└── e2e/            # 端到端测试：完整 pipeline 运行，含 WebRTC/REST API 验证
    └── python/

### 4.3 各层测试目标

#### 单元测试（无 GPU 依赖）

| 测试目标 | 方法 |
|---|---|
| `BoundedQueue` 入队/出队/溢出策略 | 纯 C++ 逻辑，Google Test |
| `ModelRegistry` SHA-256 计算、引用计数、TTL 清理 | Mock `IModelEngine`，不加载真实模型 |
| `PipelineManager` 状态机（INIT→RUNNING→DRAINING→STOPPED） | Mock Pipeline |
| `ControlChannel` ROI 坐标归一化/反归一化 | 纯数学逻辑 |
| Python DSL 节点图构建、YAML 导出/导入 | pytest，无 C++ 依赖 |
| pydantic 管理 API 数据模型校验 | pytest |

#### 集成测试（需 GPU）

| 测试目标 | 方法 |
|---|---|
| `TrtInferNode` 端到端推理（输入 tensor → 输出 bbox） | 加载真实 YOLOv8 TRT engine |
| 多 worker 并行推理结果一致性 | 对比 worker=1 和 worker=3 结果差异 <1e-4 |
| `NvDecSource` 解码帧数与视频帧数匹配 | 固定测试视频文件（100帧）|
| Pipeline 优雅停止：DRAINING 期间不丢已入队帧 | `BLOCK` 策略下计帧 |
| ModelRegistry 跨 Pipeline 共享：两条 pipeline 共用同一 engine，VRAM 增量为零 | `nvml` 查询显存 |
| `set_param()` 热更 ROI 原子性：热更后第 N+1 帧生效，第 N 帧不受影响 | 帧级断言 |

#### E2E 测试

| 测试目标 | 方法 |
|---|---|
| REST API 创建 → 启动 → 查询 → 停止 pipeline 全流程 | pytest + httpx |
| WebRTC 流建立、收到视频帧 | puppeteer / playwright 无头浏览器 |
| WebSocket ROI 热更后检测结果区域变化 | 构造测试帧，断言输出 bbox 在 ROI 内 |
| 多 Pipeline 并发：物理+化学同时运行，结果互不污染 | 两路不同测试视频 + 断言类别集合不相交 |
| 进程收到 `SIGTERM` 后所有 pipeline 优雅退出 | subprocess + 计时断言 <2s |

### 4.4 性能基准测试（非 CI gate，定期运行）

| 指标 | 目标值 | 测试方法 |
|---|---|---|
| 单路 1080p YOLOv8 吞吐 | ≥25 FPS（RTX 3090） | 跑 300 帧取均值 |
| 16路 1080p 同卡总吞吐 | ≥200 FPS | 16 路并发 |
| Pipeline 启动耗时（模型已缓存） | <500ms | 计时 |
| 优雅停止耗时 | <500ms | 计时 |
| ROI 热更生效延迟 | ≤1 帧（@25fps = 40ms） | 帧级断言 |
| GPU 显存占用（16路，共享模型） | 对比不共享减少 ≥30% | nvml 采样 |

---

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
│  │  NvDecSource              TrtInferNode                        │    │
│  │  FileSource               (parallel_workers=N)               │    │
│  │  RtspSource               Worker0: IExecContext+CudaStream   │    │
│  │                           Worker1: IExecContext+CudaStream   │    │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    HAL 硬件抽象层                               │  │
│  │  IModelEngine   IExecContext   IAllocator   ICodec             │  │
│  │       │               │            │           │               │  │
│  │  TrtEngine     TrtExecCtx    CudaAlloc    NvDecCodec           │  │
│  │  AscendEngine  AscendExecCtx AclAlloc     (二期) AscendCodec   │  │
│  │  RknnEngine    RknnExecCtx   RknnAlloc    (三期)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 核心模块说明

#### 5.2.1 节点基类体系

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

#### 5.2.2 Frame / Tensor 数据结构

```cpp
struct Frame {
    int64_t  stream_id;
    int64_t  frame_id;          // 全局单调递增，用于重排序
    int64_t  pts_us;            // 时间戳（微秒）
    Tensor   image;             // GPU / CPU tensor，含 IAllocator 管理的内存
    std::vector<Detection> detections;
    std::vector<Track>     tracks;
    std::any               user_data;  // Python 业务节点附加数据

    // 序列化钩子（预留，用于未来跨进程/跨机传输）
    std::vector<uint8_t> serialize() const;
    static Frame deserialize(const uint8_t* data, size_t len);
};

```

#### 5.2.3 BoundedQueue

```cpp
template<typename T>
class BoundedQueue {
public:
    enum class OverflowPolicy { DROP_OLDEST, DROP_NEWEST, BLOCK };

    BoundedQueue(size_t capacity, OverflowPolicy policy);
    void push(T item);          // 生产者调用
    std::optional<T> pop();     // 消费者调用（非阻塞）
    T pop_blocking();           // 消费者调用（阻塞）
    QueueStats stats() const;   // 占用率、丢帧计数、FPS
};

```

#### 5.2.4 ModelRegistry

```cpp
class ModelRegistry {
public:
    static ModelRegistry& instance();
    std::shared_ptr<IModelEngine> acquire(const std::string& engine_path);
    void release(const std::string& key);

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

```

#### 5.2.5 PipelineManager

```cpp
class PipelineManager {
public:
    std::string create(const PipelineConfig& cfg);   // 返回 pipeline_id
    void start(const std::string& id);
    void stop(const std::string& id);                // 触发 DRAINING
    void destroy(const std::string& id);
    PipelineStatus status(const std::string& id) const;
    std::vector<PipelineInfo> list() const;
};

enum class PipelineStatus { INIT, RUNNING, DRAINING, STOPPED, ERROR };

```

#### 5.2.6 HAL 接口

```cpp
// 重资源：ModelRegistry 管理，跨 pipeline 共享
class IModelEngine {
public:
    virtual std::unique_ptr<IExecContext> create_context() = 0;
    virtual size_t device_memory_bytes() const = 0;
    virtual ~IModelEngine() = default;
};

// 轻资源：每个 InferNode worker 独占
class IExecContext {
public:
    virtual void infer(const Tensor& input, Tensor& output) = 0;
    virtual ~IExecContext() = default;
};

// 设备内存分配器
class IAllocator {
public:
    virtual void* alloc(size_t bytes) = 0;
    virtual void  free(void* ptr) = 0;
    virtual MemoryType type() const = 0;  // CUDA_DEVICE / CUDA_HOST / ACL / RKNN_DMA
    virtual ~IAllocator() = default;
};

```

#### 5.2.7 ControlChannel（管理 API + 参数热更）

- 内嵌 aiohttp 协程服务，与 C++ Pipeline 同进程运行，监听独立端口（默认 8080）
- REST 接口：POST /pipelines、GET /pipelines、DELETE /pipelines/{id}、POST /pipelines/{id}/params、GET /pipelines/{id}/health
- WebSocket 接口 /ws/{pipeline_id}/control：接收 ROI 坐标、阈值等实时参数
- C++ 层 NodeBase::set_param(name, value) 使用 std::atomic 或 double-buffer，下一帧原子生效

---

## 6. 项目排期

### 6.1 阶段划分总览

| 阶段 | 目标 | 周期 |
| :--- | :--- | :--- |
| Phase 0 | 工程骨架 + CI 基础 | 第 1-2 周 |
| Phase 1 | C++ 核心调度框架 | 第 3-5 周 |
| Phase 2 | NVIDIA 推理 + 编解码 | 第 6-9 周 |
| Phase 3 | Python 绑定 + DSL | 第 10-12 周 |
| Phase 4 | 管理 API + 前端交付 | 第 13-15 周 |
| Phase 5 | 集成测试 + 性能调优 | 第 16-18 周 |

---
#### Phase 0：工程骨架 + CI 基础（第 1-2 周）

目的：建立可编译、可测试的项目骨架，CI 从第一天起运行。

任务 0.1：目录结构与 CMake 配置

- 修改文件列表
  - CMakeLists.txt（根）
  - src/core/CMakeLists.txt
  - src/hal/CMakeLists.txt
  - python/CMakeLists.txt
  - tests/CMakeLists.txt
  - .github/workflows/ci.yml
- 实现的类/函数
  - CMake targets：visionpipe_core（静态库）、visionpipe_python（nanobind 扩展）
  - FetchContent 引入：spdlog、nlohmann-json、googletest、nanobind
- 验收标准
  - cmake -B build && cmake --build build 零错误
  - ctest --test-dir build 运行空测试套件，0 failed
- 测试方法
  - CI yml 跑 cmake build + ctest

任务 0.2：基础数据结构 + 单元测试框架

- 修改文件列表
  - src/core/frame.h / frame.cpp
  - src/core/tensor.h
  - src/core/bounded_queue.h
  - tests/unit/cpp/test_bounded_queue.cpp
- 实现的类/函数
  - struct Frame（stream_id, frame_id, pts_us, user_data）
  - struct Tensor（shape, dtype, void* data, IAllocator*）
  - class BoundedQueue<T>（DROP_OLDEST / DROP_NEWEST / BLOCK）
  - struct QueueStats
- 验收标准
  - BoundedQueue 单元测试全绿：入队/出队/DROP_OLDEST 溢出/BLOCK 阻塞-唤醒
  - 覆盖率 >90%（bounded_queue.h）
- 测试方法
  - Google Test，无 GPU 依赖，CI 可直接运行

任务 0.3：日志系统初始化

- 修改文件列表
  - src/core/logger.h / logger.cpp
- 实现的类/函数
  - Logger::init(level, format)（支持 text / json 两种格式）
  - VP_LOG_INFO / VP_LOG_WARN / VP_LOG_ERROR 宏
- 验收标准
  - 单测：json 格式输出可被 nlohmann::json::parse 解析
  - CI 通过
- 测试方法
  - Google Test 解析日志输出

---
#### Phase 1：C++ 核心调度框架（第 3-5 周）

目的：实现节点图、调度器、Pipeline 生命周期，可在无 GPU 环境运行（用 Mock 节点）。

任务 1.1：节点基类与 DAG

- 修改文件列表
  - src/core/node_base.h / node_base.cpp
  - src/core/pipeline.h / pipeline.cpp
  - src/core/pipeline_builder.h
  - tests/unit/cpp/test_pipeline_dag.cpp
- 实现的类/函数
  - class NodeBase：process(Frame&)、set_param(name, val)、input_queue()、output_queue()
  - class Pipeline：add_node()、connect(a, b)、start()、stop()、状态机
  - class PipelineBuilder：>> 运算符重载
- 验收标准
  - Mock 节点组成的 DAG（Source→Filter→Sink）能跑 1000 帧，无丢帧（BLOCK 模式）
  - stop() 调用后所有节点线程在 1s 内退出
- 测试方法
  - Google Test + Mock 节点，无 GPU

任务 1.2：PipelineManager + 生命周期 API

- 修改文件列表
  - src/core/pipeline_manager.h / pipeline_manager.cpp
  - tests/unit/cpp/test_pipeline_manager.cpp
- 实现的类/函数
  - class PipelineManager：create/start/stop/destroy/status/list
  - enum class PipelineStatus
- 验收标准
  - 单测：同时创建 5 条 Mock pipeline，各自独立运行，全部优雅停止
  - stop() 触发 DRAINING，DRAINING 期间已入队帧全部处理完再退出
- 测试方法
  - Google Test，计帧断言

任务 1.3：ModelRegistry（Mock 引擎）

- 修改文件列表
  - src/core/model_registry.h / model_registry.cpp
  - src/hal/imodel_engine.h
  - tests/unit/cpp/test_model_registry.cpp
- 实现的类/函数
  - class ModelRegistry：acquire/release/gc_loop
  - std::string sha256_file(path)
  - class IModelEngine（纯虚接口）
  - class IExecContext（纯虚接口）
  - class MockModelEngine（测试用）
- 验收标准
  - 同一文件 acquire 两次：ref_count=2，只加载一次（MockEngine 构造计数=1）
  - release 两次后 ref_count=0；TTL（测试设为 100ms）到期后 engine 被销毁
  - 不同文件 acquire：各自独立实例
- 测试方法
  - Google Test，MockEngine 记录构造/析构次数

任务 1.4：parallel_workers 支持

- 修改文件列表
  - src/core/node_base.h / node_base.cpp
  - src/core/infer_node.h / infer_node.cpp
  - tests/unit/cpp/test_parallel_workers.cpp
- 实现的类/函数
  - InferNode(engine, workers=1)：启动 N 个 worker 线程
  - 输入端 work-stealing，输出端按 frame_id 重排序后入下游队列
- 验收标准
  - workers=3 时，吞吐量 ≥ workers=1 的 2.5 倍（Mock sleep 模拟推理耗时）
  - 输出帧顺序与输入一致（frame_id 严格单调递增）
- 测试方法
  - Google Test + 计时 + 帧序断言

---
#### Phase 2：NVIDIA 推理 + 编解码（第 6-9 周）

目的：接入真实 GPU，完成 TRT 推理、NVDEC/CPU 解码、YOLOv8/分类/分割验证。

任务 2.1：HAL NVIDIA 实现

- 修改文件列表
  - src/hal/nvidia/trt_model_engine.h / .cpp
  - src/hal/nvidia/trt_exec_context.h / .cpp
  - src/hal/nvidia/cuda_allocator.h / .cpp
  - tests/integration/cpp/test_trt_engine.cpp
- 实现的类/函数
  - class TrtModelEngine : public IModelEngine（加载 .engine，create_context()）
  - class TrtExecContext : public IExecContext（infer()，独立 CUDA stream）
  - class CudaAllocator : public IAllocator（cudaMalloc/cudaFree）
- 验收标准
  - 加载 YOLOv8 TRT engine，单张 1080p 推理延迟 <20ms（RTX 3090）
  - 两个 TrtExecContext 从同一 TrtModelEngine 创建，并发推理结果一致
- 测试方法
  - Google Test 集成测试，需真实 GPU

任务 2.2：视频源节点（NVDEC + CPU 软解）

- 修改文件列表
  - src/nodes/source/rtsp_source.h / .cpp
  - src/nodes/source/file_source.h / .cpp
  - src/hal/nvidia/nvdec_codec.h / .cpp
  - tests/integration/cpp/test_source_nodes.cpp
- 实现的类/函数
  - class RtspSource : public NodeBase
  - class FileSource : public NodeBase
  - class NvDecCodec : public ICodec（NVDEC 硬解）
  - CPU 软解路径（FFmpeg avcodec，作为 fallback）
- 验收标准
  - FileSource 读取 100 帧测试视频，输出恰好 100 帧，无丢帧（BLOCK 模式）
  - NVDEC 解码帧与 FFmpeg CPU 解码帧 PSNR >40dB
- 测试方法
  - 集成测试，固定测试视频文件

任务 2.3：YOLOv8 检测节点（P0）

- 修改文件列表
  - src/nodes/infer/detector_node.h / .cpp
  - src/nodes/infer/pre/letterbox_resize.h
  - src/nodes/infer/post/detection_decoder.h / .cpp
  - models/yolov8/convert.sh（ONNX→TRT 转换脚本）
  - tests/integration/cpp/test_detector_node.cpp
- 实现的类/函数
  - class DetectorNode : public InferNode
  - class LetterboxResize（CUDA kernel）
  - class DetectionDecoder（anchor-free NMS）
  - struct Detection（bbox, class_id, confidence）
- 验收标准
  - COCO val2017 subset（100张）mAP@0.5 ≥ 原始 PyTorch 结果 -1%
  - 单路 1080p ≥ 25 FPS
- 测试方法
  - 集成测试 + benchmark 脚本

任务 2.4：分类节点 + 帧内 batch（P0）

- 修改文件列表
  - src/nodes/infer/classifier_node.h / .cpp
  - src/nodes/infer/post/classification_softmax.h
  - tests/integration/cpp/test_classifier_node.cpp
- 实现的类/函数
  - class ClassifierNode : public InferNode（自动帧内 batch crop）
  - class ClassificationSoftmax
- 验收标准
  - 单帧 20 个 crop 打包成 batch=20 推理，吞吐 ≥ 单张循环推理 10×
- 测试方法
  - 集成测试，计时对比

任务 2.5：YOLOv8-seg 分割节点 + ByteTrack（P1）

- 修改文件列表
  - src/nodes/infer/segment_node.h / .cpp
  - src/nodes/infer/post/seg_mask_decoder.h
  - src/nodes/tracker/bytetrack_node.h / .cpp
  - tests/integration/cpp/test_segment_tracker.cpp
- 实现的类/函数
  - class SegmentNode : public InferNode
  - class SegMaskDecoder
  - class ByteTrackNode : public NodeBase（纯 CPU，C++ 实现）
  - struct Track（track_id, bbox, age）
- 验收标准
  - 分割 mask 与检测 bbox IOU >0.9
  - ByteTrack 在标准测试序列 MOTA >0.6
- 测试方法
  - 集成测试

---
#### Phase 3：Python 绑定 + DSL（第 10-12 周）

目的：Python 层可编排和运行完整 pipeline，用户能写自定义业务节点。

任务 3.1：nanobind 绑定核心类

- 修改文件列表
  - python/bindings/bind_pipeline.cpp
  - python/bindings/bind_nodes.cpp
  - python/bindings/bind_frame.cpp
  - python/visionpipe/__init__.py
  - tests/unit/python/test_bindings.py
- 实现的类/函数
  - 绑定：Pipeline、PipelineManager、Frame、Detection、Track
  - 绑定：RtspSource、FileSource、DetectorNode、ClassifierNode、SegmentNode、ByteTrackNode、WebRTCSink、JsonResultSink、MjpegSink
  - >> 运算符 Python 侧重载
- 验收标准
  - Python 中 src >> det >> sink; pipe.run() 能运行完整 pipeline
  - Frame 对象可在 Python 中读取 detections 列表
- 测试方法
  - pytest，集成测试需 GPU

任务 3.2：PyNode 自定义业务节点

- 修改文件列表
  - src/core/py_node.h / py_node.cpp
  - python/visionpipe/py_node.py
  - tests/unit/python/test_py_node.py
- 实现的类/函数
  - class PyNode（C++ 端，nanobind 回调 Python process 方法）
  - Python 基类 class PyNode（用户继承，重写 process(frame: Frame) -> Frame）
  - GIL acquire/release 正确处理
- 验收标准
  - 自定义 PyNode 能修改 frame.user_data 并传递到下游
  - PyNode 中抛异常不 crash C++ 线程，异常被捕获并记录日志
- 测试方法
  - pytest，Mock C++ pipeline

任务 3.3：YAML 导出/导入

- 修改文件列表
  - python/visionpipe/serialization.py
  - tests/unit/python/test_yaml_serialization.py
- 实现的类/函数
  - Pipeline.export_yaml(path) / Pipeline.load_yaml(path)
  - pydantic 模型：PipelineSpec、NodeSpec、EdgeSpec
- 验收标准
  - Python DSL 构建的 pipeline export → YAML → load，再次运行结果与原始一致
  - YAML 格式校验（pydantic）拦截非法节点类型
- 测试方法
  - pytest，无 GPU

---
#### Phase 4：管理 API + 前端交付（第 13-15 周）

目的：完成 REST 管理 API、WebRTC 视频流、WebSocket 控制通道、ROI 热更、结构化结果推送。

任务 4.1：内嵌管理 REST API

- 修改文件列表
  - python/visionpipe/server/management_api.py
  - python/visionpipe/server/schemas.py
  - tests/integration/python/test_management_api.py
- 实现的类/函数
  - POST /pipelines（body: YAML 或 JSON pipeline spec）
  - GET /pipelines
  - DELETE /pipelines/{id}
  - GET /pipelines/{id}/health（返回各节点 QueueStats）
  - POST /pipelines/{id}/params（body: {node_id, param_name, value}）
- 验收标准
  - E2E 测试：HTTP 创建→启动→查询→停止全流程 200 OK
  - health 接口返回正确的 FPS 和队列占用率
- 测试方法
  - pytest + httpx，需 GPU

任务 4.2：WebRTC Sink

- 修改文件列表
  - src/nodes/sink/webrtc_sink.h / webrtc_sink.cpp
  - python/visionpipe/server/signaling.py
  - tests/e2e/test_webrtc_stream.py
- 实现的类/函数
  - class WebRTCSink : public NodeBase（libdatachannel，NVENC H.264）
  - Python signaling server（SDP offer/answer via WebSocket）
- 验收标准
  - 浏览器（Chrome/Firefox）能打开页面看到实时视频流
  - 端到端延迟（局域网）<300ms
- 测试方法
  - Playwright 无头浏览器 E2E 测试

任务 4.3：WebSocket 控制通道 + ROI 热更

- 修改文件列表
  - python/visionpipe/server/control_ws.py
  - src/nodes/infer/detector_node.cpp（set_param ROI 实现）
  - tests/integration/python/test_roi_hotupdate.py
- 实现的类/函数
  - WebSocket endpoint /ws/{pipeline_id}/control
  - ROI 消息协议：{type: "roi", polygons: [[x,y], ...], coord: "normalized"}
  - DetectorNode::set_param("roi", polygons) 原子写（double-buffer）
- 验收标准
  - 发送 ROI 后，下一帧（≤40ms @25fps）检测结果只含 ROI 内目标
  - 并发发送 ROI 不 crash，原子性保证
- 测试方法
  - 集成测试：构造测试帧，断言帧 N+1 输出变化

任务 4.4：JsonResultSink + MjpegSink

- 修改文件列表
  - src/nodes/sink/json_result_sink.h / .cpp
  - src/nodes/sink/mjpeg_sink.h / .cpp
  - tests/integration/cpp/test_sinks.cpp
- 实现的类/函数
  - class JsonResultSink：每帧序列化 detections/tracks → WebSocket 推送
  - class MjpegSink：JPEG 编码 → multipart HTTP stream（/mjpeg/{pipeline_id}）
- 验收标准
  - JsonResultSink 输出可被 json::parse 解析，字段完整
  - MjpegSink 在浏览器 <img> 标签可直接播放
- 测试方法
  - 集成测试

---
#### Phase 5：集成测试 + 性能调优（第 16-18 周）

目的：达成性能基准目标，完成端到端测试，文档和 demo 收尾。

任务 5.1：多 Pipeline 并发集成测试

- 修改文件列表
  - tests/e2e/test_multi_pipeline.py
- 验收标准
  - 物理实验 + 化学实验同时运行，各自结果类别集合不相交
  - 共享同一 YOLOv8 backbone，VRAM 增量 ≤ 单 pipeline 的 10%
- 测试方法
  - pytest E2E，nvml 监控 VRAM

任务 5.2：性能 benchmark + 调优

- 修改文件列表
  - benchmarks/benchmark_throughput.py
  - benchmarks/benchmark_latency.py
- 验收标准（RTX 3090）

| 指标                        | 目标            |
|-----------------------------|-----------------|
| 单路 1080p YOLOv8           | ≥25 FPS         |
| 16路 1080p 同卡             | ≥200 FPS 总吞吐 |
| Pipeline 启动（模型已缓存） | <500ms          |
| 优雅停止                    | <500ms          |
| ROI 热更生效                | ≤1 帧           |

- 测试方法
  - 独立 benchmark 脚本，输出 JSON 报告

任务 5.3：文档与 Demo

- 修改文件列表
  - README.md
  - examples/quickstart.py
  - examples/multi_pipeline_demo.py
  - docs/api_reference.md
- 验收标准
  - 新用户按 README 操作，10 分钟内跑通 quickstart.py
  - multi_pipeline_demo.py 演示两个场景并发运行

---
### 6.2 项目跟踪表

| ID | 任务 | 阶段 | 优先级 | 状态 | 依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| T0.1 | 目录结构与 CMake 配置 | P0 | P0 | 待开始 | — |
| T0.2 | 基础数据结构 + 单元测试框架 | P0 | P0 | 待开始 | T0.1 |
| T0.3 | 日志系统初始化 | P0 | P0 | 待开始 | T0.1 |
| T1.1 | 节点基类与 DAG | P1 | P0 | 待开始 | T0.2 |
| T1.2 | PipelineManager + 生命周期 | P1 | P0 | 待开始 | T1.1 |
| T1.3 | ModelRegistry（Mock 引擎） | P1 | P0 | 待开始 | T0.2 |
| T1.4 | parallel_workers 支持 | P1 | P0 | 待开始 | T1.1 |
| T2.1 | HAL NVIDIA 实现（TRT） | P2 | P0 | 待开始 | T1.3 |
| T2.2 | 视频源节点（NVDEC + CPU） | P2 | P0 | 待开始 | T1.1 |
| T2.3 | YOLOv8 检测节点 | P2 | P0 | 待开始 | T2.1、T2.2 |
| T2.4 | 分类节点 + 帧内 batch | P2 | P0 | 待开始 | T2.1 |
| T2.5 | 分割节点 + ByteTrack | P2 | P1 | 待开始 | T2.1 |
| T3.1 | nanobind 绑定核心类 | P3 | P0 | 待开始 | T2.3、T2.4 |
| T3.2 | PyNode 自定义业务节点 | P3 | P0 | 待开始 | T3.1 |
| T3.3 | YAML 导出/导入 | P3 | P1 | 待开始 | T3.1 |
| T4.1 | 内嵌管理 REST API | P4 | P0 | 待开始 | T3.1 |
| T4.2 | WebRTC Sink | P4 | P0 | 待开始 | T3.1 |
| T4.3 | WebSocket 控制通道 + ROI 热更 | P4 | P0 | 待开始 | T4.1、T4.2 |
| T4.4 | JsonResultSink + MjpegSink | P4 | P0 | 待开始 | T3.1 |
| T5.1 | 多 Pipeline 并发集成测试 | P5 | P0 | 待开始 | T4.1 |
| T5.2 | 性能 benchmark + 调优 | P5 | P0 | 待开始 | T5.1 |
| T5.3 | 文档与 Demo | P5 | P1 | 待开始 | T5.2 |
