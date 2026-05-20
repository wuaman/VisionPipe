## 6. 项目排期

### 6.1 阶段划分总览

| 阶段 | 目标 | 周期 |
| :--- | :--- | :--- |
| Phase 0 | 工程骨架 + CI 基础 | 第 1-2 周 |
| Phase 1 | C++ 核心调度框架 | 第 3-5 周 |
| Phase 2 | NVIDIA 推理 + 编解码 | 第 6-9 周 |
| Phase 3 | Python 绑定 + DSL | 第 10-12 周 |
| Phase 4 | 管理 API + 前端交付 | 第 13-15 周 |
| Phase 5 | 集成验证 + 收尾 | 第 16-18 周 |

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
  - struct Frame（stream_id, frame_id, pts_us, image, detections, classifications, segments/masks, tracks, user_data: map<string, any>）
  - struct Detection（bbox, class_id, confidence, track_id）
  - struct Classification（detection_index, class_id, confidence）
  - struct Track（track_id, class_id, bbox, age, confidence）
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
  - src/core/source_node.h / source_node.cpp（新增 SourceNode 中间抽象类）
  - src/core/pipeline.h / pipeline.cpp
  - src/core/pipeline_builder.h
  - tests/unit/cpp/test_pipeline_dag.cpp
- 实现的类/函数
  - class NodeBase：name, state, queues, stats, set_param(name, val)、生命周期管理
  - class SourceNode : public NodeBase（抽象中间类）：source_worker_loop()、SourceConfig
  - class ProcessNode : public NodeBase（抽象中间类）：process(Frame&)、worker_loop + input_queue
  - class SinkNode : public NodeBase（抽象中间类）：只读消费
  - class Pipeline：add_node()、connect(a, b)、start()、stop()、状态机
  - class PipelineBuilder：>> 运算符重载
  - 合并拓扑：多个 Source 的 output_queue 可指向同一个下游 input_queue（多生产者单消费者）
  - 停止机制：队列停止级联（Source 结束 → output_queue_.stop() → 下游检测 stopped_and_empty() → 自停 → 级联）
  - 合并场景停止：Pipeline 追踪 source→queue 映射，所有 Source 结束后才 stop 共享队列
- 验收标准
  - Mock 节点组成的线性链（Source→Filter→Sink）能跑 1000 帧，无丢帧（BLOCK 模式）
  - 合并拓扑：3 个 MockSource → 1 个 MockSink，各 Source 的帧通过 stream_id 区分，全部到达 Sink
  - stop() 调用后所有节点线程在 1s 内退出
  - 合并场景停止：Source1 先结束不影响 Source2 继续 push
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
  - InferNode(engine, workers=1, max_batch_size=1, batch_timeout_ms=5)：启动 N 个 worker 线程
  - 每个 worker 动态攒帧：取到第一帧后非阻塞继续取，直到 max_batch_size 或 batch_timeout
  - 子类实现 `virtual void process_batch(std::vector<Frame>& frames) = 0`
  - InferNode 提供 `void run_inference(const Tensor& input, Tensor& output)` 辅助方法（管理 context/stream）
  - 输出端按 frame_id 重排序后入下游队列
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
  - class FileSource : public SourceNode（接受 SourceConfig）
  - class RtspSource : public SourceNode（接受 SourceConfig）
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
  - struct Classification（detection_index, class_id, confidence）
  - struct ClassifierConfig（engine_path, target_classes, max_batch_size, ...）
- ClassifierNode 双模式
  - 模式 1（二级分类）：target_classes 非空，筛选 frame.detections 中匹配类别的 bbox → crop → batch 推理 → 结果写入 frame.classifications。若无匹配的 detection，透传。
  - 模式 2（整图分类）：target_classes 为空，直接用 frame.image 整图推理 → 结果写入 frame.classifications（detection_index = -1），不依赖 detections。
- Frame 输出约定
  - ClassifierNode 结果写入 `frame.classifications`（独立字段），不覆盖 detections
  - Classification 通过 detection_index 关联对应的 detection
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

目的：Python 层可编排和运行完整 pipeline，用户能写自定义业务节点（独立进程，真并行）。

##### 设计决策（已确认）

**3-A 绑定粒度**：绑定所有核心类 + PipelineBuilder、PipelineConfig、PipelineStats、NodeStats、QueueStats、AnnotatorNode、MockModelEngine。

**3-B GIL 管理策略**：
- `Pipeline.run()` / `stop()` 释放 GIL（`call_guard<gil_scoped_release>`）
- PyNode C++ 回调时获取 GIL（`gil_scoped_acquire`）
- Frame 传递给 Python 时用 `rv_policy::reference`（零拷贝引用）

**3-C DSL 设计**：
- `>>` 直接返回 Pipeline 对象（去掉 `.build()` 步骤）
- 合并语法：`[src1, src2] >> det` 表示多 source 合并到同一下游
- 公开 API：`run(block=False, **config)` + `stop()`
  - `run(block=False)`：启动后台运行，立刻返回（默认，适合 Web 服务 / RTSP 流）
  - `run(block=True)`：启动并阻塞直到所有 source 自然结束（适合脚本 / 文件视频）
  - `stop(drain=True)`：优雅停止
- PipelineConfig 通过 Pipeline 属性或 `run()` 参数传入

**3-D CustomNode 子进程架构**：
- 用户面向 `CustomNode` 基类，重写 `on_frame(frame: FrameView)`
- `FrameView` 安全视图，process 结束后自动失效（防止悬垂引用）
- 默认 `process_mode = "subprocess"`：每个 CustomNode 跑独立进程（独立 GIL，真并行）
- 可选 `process_mode = "inline"`：同进程回调（极轻量逻辑，省 IPC 开销）
- C++ 侧新增 `ProcessProxyNode`（继承 NodeBase），负责 IPC 通道管理
- IPC 通信：Unix Domain Socket 传 metadata + CUDA IPC 零拷贝共享 GPU tensor
- Frame 始终留在 C++ 主进程，只传 metadata 副本给子进程
- 提供 `setup()` / `teardown()` 生命周期钩子
- 子进程崩溃自动重启，不影响主 pipeline

**3-E YAML 序列化**：
- `pipeline.export_yaml(path)` / `Pipeline.load_yaml(path)`
- CustomNode 序列化：记录 `module`、`class`、`process_mode`、用户自定义 config
- `load_yaml` 可自动 import 模块、实例化 CustomNode

##### 任务列表

任务 3.1：nanobind 绑定核心类 + DSL 重构

- 修改文件列表
  - python/bindings/bind_pipeline.cpp
  - python/bindings/bind_nodes.cpp
  - python/bindings/bind_frame.cpp
  - python/visionpipe/__init__.py
  - tests/unit/python/test_bindings.py
  - tests/unit/python/test_dsl.py
- 实现的类/函数
  - 绑定：Pipeline、PipelineManager、Frame、Detection、Track、所有 Node 类型、Config 类型
  - `>>` 运算符：`NodeBase.__rshift__` 直接返回 Pipeline（非 PipelineBuilder）
  - 合并语法：`list.__rshift__` 或 Pipeline 接受 list 参数，`[src1, src2] >> det` 创建合并拓扑
  - `Pipeline.run(block=False, **config)`：统一入口，`block=False` 后台运行，`block=True` 阻塞
  - `Pipeline.stop(drain=True)`：优雅停止
  - PipelineConfig 通过 Pipeline 属性或 `run()` kwargs 传入
- 验收标准
  - `([src] >> det >> sink).run()` 能运行完整 pipeline
  - `[src1, src2] >> det >> sink` 合并拓扑正确建立
  - `run(block=True)` 文件视频跑完自动返回
  - `run(block=False)` 立刻返回，`stop()` 优雅停止
  - Frame 对象可在 Python 中读取 detections 列表
- 测试方法
  - pytest，集成测试需 GPU

任务 3.2：CustomNode 子进程架构

- 修改文件列表
  - src/core/process_proxy_node.h / process_proxy_node.cpp
  - python/visionpipe/custom_node.py
  - python/visionpipe/frame_view.py
  - python/visionpipe/ipc/worker.py
  - python/visionpipe/ipc/protocol.py
  - tests/unit/python/test_custom_node.py
  - tests/integration/python/test_subprocess_node.py
- 实现的类/函数
  - C++ `ProcessProxyNode`：继承 NodeBase，IPC 通道管理，Frame metadata 序列化/反序列化
  - Python `CustomNode` 基类：`on_frame(frame: FrameView)`、`setup()`、`teardown()`
  - `FrameView`：安全视图，process 结束后自动失效
  - `Config` 内部类：`process_mode`（subprocess/inline）
  - IPC worker loop：子进程端接收 metadata + CUDA IPC handle，调用 `on_frame`，返回修改
  - 子进程崩溃检测 + 自动重启
- 验收标准
  - subprocess 模式：CustomNode 在独立进程运行，修改 user_data 传回主进程
  - inline 模式：同进程回调，行为与旧 PyNode 一致
  - 多个 subprocess CustomNode 真正并行（不受 GIL 限制）
  - 子进程抛异常不 crash 主 pipeline，异常被捕获并记录日志
  - 子进程被 kill 后自动重启，pipeline 继续运行
  - FrameView 在 on_frame 外访问时抛出 RuntimeError
- 测试方法
  - pytest + multiprocessing，需 GPU（CUDA IPC 测试）

任务 3.3：YAML 导出/导入 + CustomNode 支持

- 修改文件列表
  - python/visionpipe/serialization.py
  - tests/unit/python/test_yaml_serialization.py
- 实现的类/函数
  - Pipeline.export_yaml(path) / Pipeline.load_yaml(path)
  - pydantic 模型：PipelineSpec、NodeSpec、EdgeSpec
  - CustomNode 序列化：NodeSpec 增加 `module`、`class_name`、`process_mode` 字段
  - load_yaml 自动 import 模块、实例化 CustomNode
- 验收标准
  - Python DSL 构建的 pipeline export → YAML → load，再次运行结果与原始一致
  - YAML 格式校验（pydantic）拦截非法节点类型
  - CustomNode 可通过 YAML 自动加载（指定 module + class）
- 测试方法
  - pytest，无 GPU（序列化逻辑不需要 GPU）

---
#### Phase 4：管理 API + 前端交付（第 13-15 周）

目的：完成 REST 管理 API（分离生命周期）、WebRTC 视频流、通用 WebSocket 控制通道、ROI 热更、结构化结果推送（独立 WS）、节点状态监控接口。

任务 4.1：内嵌管理 REST API

- 修改文件列表
  - python/visionpipe/server/management_api.py
  - python/visionpipe/server/schemas.py
  - tests/integration/python/test_management_api.py
- 实现的类/函数
  - POST /pipelines（body: YAML 或 JSON pipeline spec）— 创建，状态=CREATED
  - POST /pipelines/{id}/start — 启动
  - POST /pipelines/{id}/stop — 停止（可重启）
  - GET /pipelines — 列表（含状态）
  - DELETE /pipelines/{id} — 销毁（必须先 stop）
  - GET /pipelines/{id}/health（返回各节点 QueueStats + FPS）
  - GET /pipelines/{id}/nodes（返回 per-node NodeStats：fps/latency_ms/frames_processed/errors/state）
  - POST /pipelines/{id}/params（body: {node_id, param_name, value}）
- 验收标准
  - E2E 测试：HTTP 创建→启动→查询→停止→销毁全流程 200 OK
  - health 接口返回正确的 FPS 和队列占用率
  - nodes 接口返回所有节点的 NodeStats，字段完整
  - 未 stop 直接 DELETE 返回 409 Conflict
- 测试方法
  - pytest + httpx，需 GPU

任务 4.2：WebRTC Sink

- 修改文件列表
  - src/nodes/sink/webrtc_sink.h / webrtc_sink.cpp
  - python/visionpipe/server/signaling.py
  - tests/e2e/test_webrtc_stream.py
- 实现的类/函数
  - class WebRTCSink : public SinkNode（libdatachannel，NVENC H.264）
  - Python signaling server（SDP offer/answer via WebSocket /ws/{id}/webrtc）
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
  - WebSocket endpoint /ws/{pipeline_id}/control（通用控制通道）
  - 消息格式：{type: "roi"|"set_param"|"ping", payload: {...}}
  - ROI payload：{polygons: [[x,y], ...], coord: "normalized"}
  - set_param payload：{node_id, param_name, value} — 转发到对应节点的 set_param()
  - DetectorNode::set_param("roi", polygons) 原子写（double-buffer）
- 验收标准
  - 发送 ROI 后，下一帧（≤40ms @25fps）检测结果只含 ROI 内目标
  - 通用 set_param 可修改任意节点参数（如 threshold、enabled）
  - 并发发送不 crash，原子性保证
- 测试方法
  - 集成测试：构造测试帧，断言帧 N+1 输出变化

任务 4.4：JsonResultSink + MjpegSink

- 修改文件列表
  - src/nodes/sink/json_result_sink.h / .cpp
  - src/nodes/sink/mjpeg_sink.h / .cpp
  - src/nodes/sink/sink_node.h（SinkNode 基类，提供 enabled 属性）
  - tests/integration/cpp/test_sinks.cpp
- 实现的类/函数
  - class SinkNode : public NodeBase（抽象中间类，统一 enabled 属性）
  - class JsonResultSink : public SinkNode — 每帧序列化 detections/tracks，通过独立 WebSocket /ws/{id}/results 推送
  - class MjpegSink : public SinkNode — JPEG 编码 → multipart HTTP stream（/mjpeg/{pipeline_id}），默认 enabled=false
- 验收标准
  - JsonResultSink 输出可被 json::parse 解析，字段完整
  - MjpegSink 默认关闭，set_param("enabled", true) 后浏览器 <img> 标签可直接播放
  - SinkNode enabled=false 时不消耗 CPU/GPU 资源（跳过处理）
- 测试方法
  - 集成测试

---
#### Phase 5：集成验证 + 收尾（第 16-18 周）

目的：确保 Phase 0-4 每个环节功能正确、数据流完整、多 pipeline 互不干扰，文档和 demo 收尾。

任务 5.1：多 Pipeline 并发集成测试

- 修改文件列表
  - tests/e2e/test_multi_pipeline.py
- 实现的类/函数
  - 两路 pipeline 并发运行测试（不同测试视频 / 不同 class filter）
  - ModelRegistry 共享验证（acquire 同一 engine 只加载一次）
  - 生命周期隔离验证（一路 stop 不影响另一路）
- 验收标准
  - 两路 pipeline 同时运行，各自结果类别集合不相交
  - ModelRegistry 模型共享生效：两路 acquire 同一 engine，构造计数 = 1
  - 一路 stop() 不影响另一路继续运行
  - 全部 stop() 后资源正常释放（无 GPU 内存泄漏）
- 测试方法
  - pytest E2E，需 GPU
- 测试代码骨架
  ```python
  # tests/e2e/test_multi_pipeline.py
  @pytest.mark.gpu
  def test_two_pipelines_disjoint_classes():
      """两路 pipeline 并发运行，结果类别集合不相交"""
      pipe_a = FileSource("video_a.mp4") >> DetectorNode(engine, class_filter=[0,1,2]) >> sink_a
      pipe_b = FileSource("video_b.mp4") >> DetectorNode(engine, class_filter=[10,11,12]) >> sink_b

      pipe_a.run(block=False)
      pipe_b.run(block=False)
      time.sleep(5)
      pipe_a.stop()
      pipe_b.stop()

      classes_a = {d.class_id for f in sink_a.frames for d in f.detections}
      classes_b = {d.class_id for f in sink_b.frames for d in f.detections}
      assert classes_a.isdisjoint(classes_b)

  @pytest.mark.gpu
  def test_shared_model_single_load():
      """共享同一 engine，ModelRegistry 只加载一次"""
      registry = ModelRegistry.instance()
      engine1 = registry.acquire("yolov8n.engine")
      engine2 = registry.acquire("yolov8n.engine")
      assert engine1 is engine2  # 同一实例

  @pytest.mark.gpu
  def test_independent_lifecycle():
      """一路停止不影响另一路"""
      pipe_a.run(block=False)
      pipe_b.run(block=False)
      pipe_a.stop()
      assert pipe_b.status == PipelineStatus.RUNNING
      pipe_b.stop()
  ```

任务 5.2：端到端验证测试

- 修改文件列表
  - tests/e2e/test_e2e_validation.py
- 实现的类/函数
  - 第一层：节点级正确性验证（每个节点输出符合预期）
  - 第二层：数据流完整性验证（Frame 累积语义正确）
  - 第三层：控制面验证（REST API / WebSocket / YAML 往返）
- 验收标准
  - **第一层 — 节点级正确性**：
    - FileSource：100 帧视频 → 输出恰好 100 帧，frame_id 单调递增，image 非空
    - DetectorNode：有目标测试图 → detections 非空，bbox ∈ [0,1]，confidence > 阈值
    - ClassifierNode 二级分类：detections 非空时 classifications 非空，detection_index 合法
    - ClassifierNode 整图分类：detection_index = -1，不依赖 detections
    - TrackerNode：连续帧相同目标 → track_id 保持一致（至少 N 帧内不变）
    - CustomNode（subprocess 模式）：user_data["test_key"] 在下游可读
    - AnnotatorNode：输出 image 尺寸与输入一致
    - JsonResultSink：输出 JSON 可解析，包含 detections/tracks 字段
  - **第二层 — 数据流完整性**：
    - 完整链路 `FileSource → DetectorNode → ClassifierNode → TrackerNode → CustomNode → AnnotatorNode → JsonResultSink` 跑通 100 帧无 crash
    - 链路末端 Frame 同时具备：stream_id/frame_id/pts_us（Source）、detections 非空且 bbox 合法（Detector）、classifications 非空且 detection_index 有效（Classifier）、tracks 非空且 track_id > 0（Tracker）、user_data["test_key"] 存在（CustomNode）
  - **第三层 — 控制面验证**：
    - REST API：create → start → GET health（FPS > 0）→ GET nodes（每个节点 state=RUNNING）→ stop → delete 全流程 200 OK
    - WebSocket 控制通道：发送 set_param → 验证节点参数生效
    - YAML 往返：DSL 构建 pipeline → export_yaml → load_yaml → 再次运行，Sink 输出结构一致
- 测试方法
  - pytest E2E，需 GPU
- 测试代码骨架
  ```python
  # tests/e2e/test_e2e_validation.py

  # ── 第一层：节点级正确性 ──

  @pytest.mark.gpu
  def test_filesource_frame_count():
      """FileSource 输出帧数与视频帧数一致"""
      src = FileSource("sample_100frames.mp4", decode_mode="auto")
      sink = CollectorSink()
      pipe = src >> sink
      pipe.run(block=True)
      assert len(sink.frames) == 100
      for i, f in enumerate(sink.frames):
          assert f.frame_id == i
          assert f.image is not None

  @pytest.mark.gpu
  def test_detector_produces_valid_detections():
      """DetectorNode 输出 bbox 在 [0,1] 范围内"""
      src = FileSource("sample_with_objects.mp4")
      det = DetectorNode(engine_path="yolov8n.engine")
      sink = CollectorSink()
      pipe = src >> det >> sink
      pipe.run(block=True)
      frames_with_dets = [f for f in sink.frames if len(f.detections) > 0]
      assert len(frames_with_dets) > 0  # 至少有目标被检测到
      for f in frames_with_dets:
          for d in f.detections:
              assert 0 <= d.bbox[0] <= 1 and 0 <= d.bbox[2] <= 1  # x 范围
              assert 0 <= d.bbox[1] <= 1 and 0 <= d.bbox[3] <= 1  # y 范围
              assert d.confidence > 0

  @pytest.mark.gpu
  def test_tracker_consistent_ids():
      """连续帧中同一目标的 track_id 保持一致"""
      pipe = src >> det >> tracker >> sink
      pipe.run(block=True)
      # 提取前 N 帧的 track_id 集合，验证存在至少一个跨帧一致的 id
      all_track_ids = [set(t.track_id for t in f.tracks) for f in sink.frames if f.tracks]
      assert len(all_track_ids) >= 2
      persistent_ids = all_track_ids[0]
      for ids in all_track_ids[1:5]:
          persistent_ids &= ids
      assert len(persistent_ids) > 0  # 至少一个目标持续被追踪

  # ── 第二层：数据流完整性 ──

  @pytest.mark.gpu
  def test_full_chain_data_integrity():
      """完整链路：每个节点的输出在 Frame 中累积"""
      src = FileSource("sample_with_objects.mp4")
      det = DetectorNode(engine_path="yolov8n.engine")
      cls = ClassifierNode(engine_path="resnet50.engine", target_classes=[0, 1])
      trk = TrackerNode()
      custom = TestCustomNode()  # subprocess, 写入 user_data["smoke"]
      ann = AnnotatorNode()
      sink = CollectorSink()

      pipe = src >> det >> cls >> trk >> custom >> ann >> sink
      pipe.run(block=True)

      for frame in sink.frames:
          # Source 写入
          assert frame.frame_id >= 0
          assert frame.image is not None
          # Detector 写入
          if len(frame.detections) > 0:
              assert all(d.bbox[2] > d.bbox[0] for d in frame.detections)
          # Classifier 写入（有 detections 时）
          if len(frame.detections) > 0:
              assert len(frame.classifications) > 0
              for c in frame.classifications:
                  assert 0 <= c.detection_index < len(frame.detections)
          # Tracker 写入
          if len(frame.detections) > 0:
              assert len(frame.tracks) > 0
              assert all(t.track_id > 0 for t in frame.tracks)
          # CustomNode 写入
          assert "smoke" in frame.user_data

  # ── 第三层：控制面验证 ──

  @pytest.mark.gpu
  async def test_rest_api_full_lifecycle():
      """REST API 全生命周期"""
      async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
          # create
          resp = await client.post("/pipelines", json=pipeline_spec)
          assert resp.status_code == 200
          pid = resp.json()["id"]
          # start
          resp = await client.post(f"/pipelines/{pid}/start")
          assert resp.status_code == 200
          # health
          resp = await client.get(f"/pipelines/{pid}/health")
          assert resp.json()["fps"] > 0
          # nodes
          resp = await client.get(f"/pipelines/{pid}/nodes")
          for node in resp.json()["nodes"]:
              assert node["state"] == "RUNNING"
          # stop
          resp = await client.post(f"/pipelines/{pid}/stop")
          assert resp.status_code == 200
          # delete
          resp = await client.delete(f"/pipelines/{pid}")
          assert resp.status_code == 200

  @pytest.mark.gpu
  def test_yaml_roundtrip():
      """YAML 往返一致性"""
      pipe = src >> det >> tracker >> sink
      pipe.export_yaml("/tmp/test_pipe.yaml")
      pipe2 = Pipeline.load_yaml("/tmp/test_pipe.yaml")
      pipe2.run(block=True)
      # 验证节点结构一致
      assert [n.name for n in pipe.nodes] == [n.name for n in pipe2.nodes]
  ```

任务 5.3：文档与 Demo

- 修改文件列表
  - README.md
  - examples/quickstart.py
  - examples/multi_pipeline_demo.py
  - docs/api_reference.md
- 验收标准
  - 新用户按 README 操作，10 分钟内跑通 quickstart.py
  - multi_pipeline_demo.py 演示两个场景并发运行
  - 基于 RTX 3060 12GB + WSL2 环境验证可运行

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
| T2.2b | ICodec HAL 抽象 + 跨平台编解码（二期） | — | P1 | 未来扩展 | T2.2a | 移至「未来扩展」章节，不纳入正式排期 |
| T2.3 | YOLOv8 检测节点 | P2 | P0 | [x] | T2.1、T2.2a |
| T2.4 | 分类节点 + 帧内 batch | P2 | P0 | [x] | T2.1 |
| T2.5 | 分割节点 + ByteTrack | P2 | P1 | [ ] | T2.1 |
| T3.1 | nanobind 绑定核心类 + DSL 重构 | P3 | P0 | [ ] | T2.3、T2.4 | 需要：>> 返回 Pipeline、合并语法、run(block)/stop() API |
| T3.2 | CustomNode 子进程架构 | P3 | P0 | [ ] | T3.1 | 需要：ProcessProxyNode + CustomNode + FrameView + IPC + subprocess |
| T3.3 | YAML 导出/导入 + CustomNode 支持 | P3 | P1 | [ ] | T3.1 | 需要：CustomNode 序列化（module/class 自动导入） |
| T4.1 | 内嵌管理 REST API | P4 | P0 | [ ] | T3.1 | 需要：分离生命周期（create/start/stop/delete）+ 新增 GET /nodes 接口 + NodeStats 补齐 latency_ms/state |
| T4.2 | WebRTC Sink | P4 | P0 | [ ] | T3.1 | 需要：继承 SinkNode（非 NodeBase） |
| T4.3 | WebSocket 控制通道 + ROI 热更 | P4 | P0 | [ ] | T4.1、T4.2 | 需要：通用 set_param 消息转发（不仅 ROI） |
| T4.4 | JsonResultSink + MjpegSink | P4 | P0 | [ ] | T3.1 | 需要：SinkNode 基类 + enabled 属性 + MjpegSink 默认关闭 + JsonResult 独立 WS |
| T5.1 | 多 Pipeline 并发集成测试 | P5 | P0 | [ ] | T4.1 | 需 rework：修复测试资产路径 + 去掉 VRAM ≤10% 硬指标 |
| T5.2 | 端到端验证测试（三层验证） | P5 | P0 | [ ] | T4.1、T4.3、T4.4 |
| T5.3 | 文档与 Demo | P5 | P1 | [ ] | T5.1、T5.2 |

---

## 6.3 未来扩展（不纳入当前排期）

#### ICodec HAL 抽象 + 跨平台编解码（原 T2.2b）

一期 SourceNode 直接调用 `cv::cudacodec`（GPU）/ `cv::VideoCapture`（CPU），满足 NVIDIA 平台需求。后续如需支持异构硬件（华为昇腾 DVPP、瑞芯微 MPP），引入 ICodec HAL 抽象层：

- `class ICodec`：纯虚接口（open / decode_next / close / device_type）
- `class NvDecCodec : public ICodec`（可升级为 FFmpeg CUVID 或直接 NVCUVID API）
- `class OpenCvCodec : public ICodec`（封装当前 cv::cudacodec + VideoCapture fallback）
- 预留：`DvppCodec`（昇腾）、`MppCodec`（瑞芯微）
- Source 节点通过 ICodec 工厂按平台选择后端，上层 API 不变

#### 性能 benchmark + 专项调优

待功能全链路验证通过后，独立做性能评测（参见 Section 4.4 参考指标），包括：
- 吞吐量 benchmark（单路 / 多路）
- 延迟 profiling（节点级 latency 分析）
- 显存优化（模型共享效率、Tensor 内存池）
- 多 pipeline 并发调度优化

---
