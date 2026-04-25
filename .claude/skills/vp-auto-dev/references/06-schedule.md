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

**⚠️ 阶段门禁**：本阶段所有任务测试通过后，方可进入 Phase 1。

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
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_cmake_build.cpp
  TEST(CMakeBuildTest, LibraryExists) {
      // 验证 visionpipe_core 静态库可链接
      EXPECT_TRUE(true);  // 占位，实际验证符号导出
  }
  ```

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
  - Google Test，需 GPU 环境
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_bounded_queue.cpp
  class BoundedQueueTest : public ::testing::Test {
  protected:
      BoundedQueue<int> queue_{10, OverflowPolicy::DROP_OLDEST};
  };

  TEST_F(BoundedQueueTest, PushPopBasic) {
      queue_.push(42);
      auto result = queue_.pop();
      ASSERT_TRUE(result.has_value());
      EXPECT_EQ(*result, 42);
  }

  TEST_F(BoundedQueueTest, DropOldestOnOverflow) {
      for (int i = 0; i < 15; ++i) queue_.push(i);
      auto stats = queue_.stats();
      EXPECT_EQ(stats.dropped_count, 5);  // 15 - 10 capacity
      EXPECT_EQ(*queue_.pop(), 5);  // 最老的 0-4 被丢弃
  }

  TEST_F(BoundedQueueTest, BlockOnFull) {
      BoundedQueue<int> block_queue(2, OverflowPolicy::BLOCK);
      block_queue.push(1);
      block_queue.push(2);
      // 异步 pop 以解除阻塞
      std::thread popper([&]() {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          block_queue.pop();
      });
      auto start = std::chrono::steady_clock::now();
      block_queue.push(3);  // 应阻塞直到 pop
      auto elapsed = std::chrono::steady_clock::now() - start;
      EXPECT_GE(elapsed, std::chrono::milliseconds(50));
      popper.join();
  }
  ```

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
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_logger.cpp
  TEST(LoggerTest, JsonFormatParsable) {
      std::stringstream ss;
      auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(ss);
      Logger::init(sink, spdlog::level::info, "json");

      VP_LOG_INFO("test message {}", 42);

      auto json = nlohmann::json::parse(ss.str());
      EXPECT_EQ(json["level"], "info");
      EXPECT_EQ(json["message"], "test message 42");
  }
  ```

---
#### Phase 1：C++ 核心调度框架（第 3-5 周）

目的：实现节点图、调度器、Pipeline 生命周期。

**⚠️ 阶段门禁**：本阶段所有任务测试通过后，方可进入 Phase 2。

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
  - Google Test + Mock 节点，需 GPU 环境
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_pipeline_dag.cpp
  class MockSource : public NodeBase {
  public:
      MockSource() { set_name("source"); }
      void process(Frame& frame) override {
          frame.frame_id = frame_counter_.fetch_add(1);
          output_queue()->push(frame);
      }
  private:
      std::atomic<int64_t> frame_counter_{0};
  };

  class MockFilter : public NodeBase {
  public:
      MockFilter() { set_name("filter"); }
      void process(Frame& frame) override {
          frame.user_data = 42;  // 添加标记
          output_queue()->push(frame);
      }
  };

  class MockSink : public NodeBase {
  public:
      MockSink() { set_name("sink"); }
      void process(Frame& frame) override {
          std::lock_guard<std::mutex> lock(mu_);
          received_frames_.push_back(frame);
      }
      const std::vector<Frame>& received() const { return received_frames_; }
  private:
      std::mutex mu_;
      std::vector<Frame> received_frames_;
  };

  TEST(PipelineDagTest, SourceFilterSinkChain) {
      auto src = std::make_shared<MockSource>();
      auto filter = std::make_shared<MockFilter>();
      auto sink = std::make_shared<MockSink>();

      Pipeline pipe;
      pipe.add_node(src);
      pipe.add_node(filter);
      pipe.add_node(sink);
      pipe.connect(src, filter);
      pipe.connect(filter, sink);

      pipe.start();
      std::this_thread::sleep_for(std::chrono::seconds(2));
      pipe.stop();

      auto& frames = sink->received();
      EXPECT_GE(frames.size(), 100);
      for (const auto& f : frames) {
          EXPECT_EQ(std::any_cast<int>(f.user_data), 42);
      }
  }
  ```

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
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_pipeline_manager.cpp
  TEST(PipelineManagerTest, MultiPipelineLifecycle) {
      PipelineManager mgr;
      std::vector<std::string> ids;

      for (int i = 0; i < 5; ++i) {
          PipelineConfig cfg;
          cfg.name = fmt::format("pipe_{}", i);
          ids.push_back(mgr.create(cfg));
      }

      for (const auto& id : ids) {
          mgr.start(id);
          EXPECT_EQ(mgr.status(id), PipelineStatus::RUNNING);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      for (const auto& id : ids) {
          mgr.stop(id);
          EXPECT_EQ(mgr.status(id), PipelineStatus::STOPPED);
      }

      for (const auto& id : ids) {
          mgr.destroy(id);
      }
      EXPECT_TRUE(mgr.list().empty());
  }

  TEST(PipelineManagerTest, DrainingCompletesQueuedFrames) {
      PipelineManager mgr;
      auto id = mgr.create(PipelineConfig{.name = "drain_test"});
      mgr.start(id);

      // 注入帧到队列
      auto& pipe = mgr.get(id);
      auto& queue = pipe->get_node("sink")->input_queue();
      for (int i = 0; i < 100; ++i) {
          Frame f;
          f.frame_id = i;
          queue->push(f);
      }

      auto before_count = pipe->processed_count();
      mgr.stop(id);  // 触发 DRAINING
      auto after_count = pipe->processed_count();

      EXPECT_EQ(after_count - before_count, 100);  // 队列中帧全部处理完
  }
  ```

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
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_model_registry.cpp
  class CountingMockEngine : public IModelEngine {
  public:
      static std::atomic<int> construct_count{0};
      static std::atomic<int> destruct_count{0};

      CountingMockEngine() { construct_count++; }
      ~CountingMockEngine() override { destruct_count++; }

      std::unique_ptr<IExecContext> create_context() override {
          return std::make_unique<MockExecContext>();
      }
      size_t device_memory_bytes() const override { return 1024; }
  };

  TEST(ModelRegistryTest, AcquireSameFileDedup) {
      CountingMockEngine::construct_count = 0;
      CountingMockEngine::destruct_count = 0;

      auto& registry = ModelRegistry::instance();
      registry.set_engine_factory([](const std::string&) {
          return std::make_shared<CountingMockEngine>();
      });
      registry.set_ttl(std::chrono::milliseconds(100));

      auto engine1 = registry.acquire("model_a.engine");
      auto engine2 = registry.acquire("model_a.engine");

      EXPECT_EQ(CountingMockEngine::construct_count, 1);  // 只构造一次
      EXPECT_EQ(engine1.get(), engine2.get());  // 同一实例

      registry.release("model_a.engine");
      registry.release("model_a.engine");

      std::this_thread::sleep_for(std::chrono::milliseconds(150));
      EXPECT_EQ(CountingMockEngine::destruct_count, 1);  // TTL 后析构
  }

  TEST(ModelRegistryTest, AcquireDifferentFiles) {
      CountingMockEngine::construct_count = 0;

      auto& registry = ModelRegistry::instance();
      auto engine1 = registry.acquire("model_a.engine");
      auto engine2 = registry.acquire("model_b.engine");

      EXPECT_EQ(CountingMockEngine::construct_count, 2);
      EXPECT_NE(engine1.get(), engine2.get());
  }
  ```

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
- 测试代码骨架
  ```cpp
  // tests/unit/cpp/test_parallel_workers.cpp
  class SlowMockEngine : public IModelEngine {
  public:
      std::unique_ptr<IExecContext> create_context() override {
          return std::make_unique<SlowMockContext>();
      }
      size_t device_memory_bytes() const override { return 1024; }

      class SlowMockContext : public IExecContext {
      public:
          void infer(const Tensor& input, Tensor& output) override {
              std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 模拟推理
          }
      };
  };

  TEST(ParallelWorkersTest, ThroughputScale) {
      auto engine = std::make_shared<SlowMockEngine>();

      // workers=1 基准
      auto node1 = std::make_shared<InferNode>(engine, 1);
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 100; ++i) {
          Frame f;
          f.frame_id = i;
          node1->input_queue()->push(f);
      }
      // 等待处理完成...
      auto elapsed_1worker = std::chrono::steady_clock::now() - start;

      // workers=3
      auto node3 = std::make_shared<InferNode>(engine, 3);
      start = std::chrono::steady_clock::now();
      for (int i = 0; i < 100; ++i) {
          Frame f;
          f.frame_id = i;
          node3->input_queue()->push(f);
      }
      // 等待处理完成...
      auto elapsed_3workers = std::chrono::steady_clock::now() - start;

      // 吞吐应接近 3x，允许一定调度开销
      EXPECT_LE(elapsed_3workers, elapsed_1worker * 100.0 / 250.0);
  }

  TEST(ParallelWorkersTest, OutputOrderPreserved) {
      auto engine = std::make_shared<SlowMockEngine>();
      auto node = std::make_shared<InferNode>(engine, 3);

      std::vector<int64_t> frame_ids;
      for (int i = 0; i < 100; ++i) {
          Frame f;
          f.frame_id = i;
          node->input_queue()->push(f);
      }

      // 从下游队列读取
      auto out_queue = node->output_queue();
      for (int i = 0; i < 100; ++i) {
          auto f = out_queue->pop_blocking();
          frame_ids.push_back(f.frame_id);
      }

      // 验证严格单调递增
      for (int i = 1; i < 100; ++i) {
          EXPECT_EQ(frame_ids[i], frame_ids[i-1] + 1);
      }
  }
  ```

---
#### Phase 2：NVIDIA 推理 + 编解码（第 6-9 周）

目的：接入真实 GPU，完成 TRT 推理、`cv::cudacodec` GPU 硬解码 / CPU 软解码、YOLOv8/分类/分割验证。

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

任务 2.2a：视频源节点（`cv::cudacodec` GPU 硬解码，一期）

- 修改文件列表
  - src/nodes/source/source_config.h
  - src/nodes/source/rtsp_source.h / .cpp
  - src/nodes/source/file_source.h / .cpp
  - tests/integration/cpp/test_source_nodes.cpp
- 实现的类/函数
  - enum class DecodeMode { AUTO, GPU, CPU }
  - struct SourceConfig（uri, decode_mode, gpu_device, queue_capacity, overflow_policy）
  - class FileSource : public NodeBase（接受 SourceConfig）
  - class RtspSource : public NodeBase（接受 SourceConfig）
  - GPU 路径：`cv::cudacodec::VideoReader::nextFrame()` → `cv::cuda::GpuMat` → Frame.image
  - CPU 路径：`cv::VideoCapture::read()` → `cv::Mat` → `GpuMat::upload()` → Frame.image
  - AUTO 模式：运行时检测 NVCUVID 可用性，优先 GPU，不可用时自动退化为 CPU 并记日志
  - GPU 模式：强制硬解，NVCUVID 不可用时抛 `CudaError`
  - ICodec 接口暂不实现，Source 节点内部直接调用 OpenCV
- 验收标准
  - FileSource 读取 100 帧测试视频，输出恰好 100 帧，无丢帧（BLOCK 模式）
  - RtspSource 能连接测试 RTSP 流并持续输出帧
  - `decode_mode=GPU`：解码帧直接在 GPU 显存，无 CPU↔GPU 拷贝
  - `decode_mode=CPU`：解码帧经 CPU → GPU upload，功能正确
  - `decode_mode=AUTO`：有 NVCUVID 时走 GPU 路径，无则走 CPU 路径
- 测试方法
  - 集成测试，固定测试视频文件，分别测试 GPU / CPU / AUTO 三种模式

任务 2.2b：ICodec HAL 抽象 + 跨平台编解码（二期，Phase 5 或独立优化迭代）

- 修改文件列表
  - src/hal/icodec.h（新增 ICodec HAL 接口）
  - src/hal/nvidia/nvdec_codec.h / .cpp
  - src/nodes/source/file_source.cpp（重构为通过 ICodec 抽象解码）
  - src/nodes/source/rtsp_source.cpp（同上）
  - CMakeLists.txt（按需新增 FFmpeg / 厂商 SDK 依赖）
  - tests/integration/cpp/test_icodec_impl.cpp
- 实现的类/函数
  - class ICodec（HAL 纯虚接口：open / decode_next / close / device_type）
  - class NvDecCodec : public ICodec（可选升级为 FFmpeg CUVID 或直接 NVCUVID API，精细控制 CUDA stream）
  - class OpenCvCodec : public ICodec（封装 `cv::cudacodec` + `cv::VideoCapture` fallback）
  - （预留）class DvppCodec : public ICodec（华为昇腾 DVPP）
  - （预留）class MppCodec : public ICodec（瑞芯微 MPP）
  - Source 节点通过 ICodec 工厂按平台选择后端
- 验收标准
  - ICodec 接口测试：至少两种实现（NvDec + OpenCv）通过同一测试套件
  - Source 节点通过 ICodec 工厂切换后端，功能不变
- 测试方法
  - 集成测试 + benchmark 对比脚本

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
  - models/resnet50/convert.sh（ONNX→TRT 转换脚本）
  - models/efficientnet_b0/convert.sh
  - models/shufflenetv2/convert.sh
  - tests/integration/cpp/test_classifier_node.cpp
- 实现的类/函数
  - class ClassifierNode : public InferNode（自动帧内 batch crop）
  - class ClassificationSoftmax
- Frame 输出约定
  - ClassifierNode 读取 `frame.detections`，对每个 detection 按 bbox 坐标从 `frame.image` 裁剪 crop
  - 所有 crop 拼成 batch=N 一次推理（N = 当前帧 detections 数量）
  - 推理完成后按 index 回写：`detections[i].class_id` ← 分类标签，`detections[i].confidence` ← softmax 最大概率
  - 不新增 Frame 字段，不修改 `frame.image`，不修改 `detections[i].bbox`
  - 若 `frame.detections` 为空，ClassifierNode 直接透传 Frame，不做任何推理
- 验收标准
  - ResNet50 / EfficientNet-B0 / ShuffleNetV2 三个模型均完成 ONNX→TRT 转换并通过推理验证
  - 单帧 20 个 crop 打包成 batch=20 推理，吞吐 ≥ 单张循环推理 10×
  - detections 为空时，Frame 原样透传，不触发推理
- 测试方法
  - 集成测试，计时对比，三个模型分别验证

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
  - 绑定：DecodeMode、SourceConfig、RtspSource、FileSource、DetectorNode、ClassifierNode、SegmentNode、ByteTrackNode、WebRTCSink、JsonResultSink、MjpegSink
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

目的：达成性能基准目标，完成端到端测试，ICodec HAL 跨平台编解码抽象（T2.2b），文档和 demo 收尾。

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
| T0.1 | 目录结构与 CMake 配置 | P0 | P0 | [x] | — |
| T0.2 | 基础数据结构 + 单元测试框架 | P0 | P0 | [x] | T0.1 |
| T0.3 | 日志系统初始化 | P0 | P0 | [x] | T0.1 |
| T1.1 | 节点基类与 DAG | P1 | P0 | [x] | T0.2 |
| T1.2 | PipelineManager + 生命周期 | P1 | P0 | [x] | T1.1 |
| T1.3 | ModelRegistry（Mock 引擎） | P1 | P0 | [x] | T0.2 |
| T1.4 | parallel_workers 支持 | P1 | P0 | [x] | T1.1 |
| T2.1 | HAL NVIDIA 实现（TRT） | P2 | P0 | [x] | T1.3 |
| T2.2a | 视频源节点（`cv::cudacodec` GPU 硬解，一期） | P2 | P0 | [x] | T1.1 |
| T2.2b | ICodec HAL 抽象 + 跨平台编解码（二期） | P5 | P1 | [ ] | T2.2a、T5.1 |
| T2.3 | YOLOv8 检测节点 | P2 | P0 | [ ] | T2.1、T2.2a |
| T2.4 | 分类节点 + 帧内 batch | P2 | P0 | [ ] | T2.1 |
| T2.5 | 分割节点 + ByteTrack | P2 | P1 | [ ] | T2.1 |
| T3.1 | nanobind 绑定核心类 | P3 | P0 | [ ] | T2.3、T2.4 |
| T3.2 | PyNode 自定义业务节点 | P3 | P0 | [ ] | T3.1 |
| T3.3 | YAML 导出/导入 | P3 | P1 | [ ] | T3.1 |
| T4.1 | 内嵌管理 REST API | P4 | P0 | [ ] | T3.1 |
| T4.2 | WebRTC Sink | P4 | P0 | [ ] | T3.1 |
| T4.3 | WebSocket 控制通道 + ROI 热更 | P4 | P0 | [ ] | T4.1、T4.2 |
| T4.4 | JsonResultSink + MjpegSink | P4 | P0 | [ ] | T3.1 |
| T5.1 | 多 Pipeline 并发集成测试 | P5 | P0 | [ ] | T4.1 |
| T5.2 | 性能 benchmark + 调优 | P5 | P0 | [ ] | T5.1 |
| T5.3 | 文档与 Demo | P5 | P1 | [ ] | T5.2 |

---
