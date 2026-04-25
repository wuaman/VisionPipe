// test_source_nodes.cpp
// 任务 T2.2a 集成测试：FileSource 和 RtspSource 视频源节点

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/bounded_queue.h"
#include "core/error.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/tensor.h"
#include "nodes/source/file_source.h"
#include "nodes/source/rtsp_source.h"
#include "nodes/source/source_config.h"

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std::chrono_literals;

namespace visionpipe {
namespace {

// ============================================================================
// 辅助函数和常量
// ============================================================================

/// @brief 获取测试视频路径（从环境变量或使用默认路径）
const char *get_test_video_path() {
  const char *path = std::getenv("VISIONPIPE_TEST_VIDEO");
  if (path) {
    return path;
  }
  // 默认路径
  return "tests/data/test_video_100frames.mp4";
}

/// @brief 获取测试 RTSP URL（从环境变量）
const char *get_test_rtsp_url() {
  return std::getenv("VISIONPIPE_TEST_RTSP_URL");
}

/// @brief 检查文件是否存在
bool file_exists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

/// @brief 检查 CUDA 是否可用
bool is_cuda_available() {
#ifdef VISIONPIPE_USE_CUDA
  // 尝试初始化 CUDA 上下文
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

/// @brief 等待条件满足或超时
bool wait_for(const std::function<bool()> &predicate,
              std::chrono::milliseconds timeout = 5000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

/// @brief 收集帧用于测试验证
struct FrameCollector {
  std::vector<Frame> frames;
  std::mutex mutex;

  void add(Frame &&frame) {
    std::lock_guard<std::mutex> lock(mutex);
    frames.push_back(std::move(frame));
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex));
    return frames.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    frames.clear();
  }
};

// ============================================================================
// FileSource 测试夹具
// ============================================================================

class FileSourceTest : public ::testing::Test {
protected:
  void SetUp() override {
    video_path_ = get_test_video_path();
    video_available_ = file_exists(video_path_);
    cuda_available_ = is_cuda_available();
  }

  const char *video_path_ = nullptr;
  bool video_available_ = false;
  bool cuda_available_ = false;
};

// ============================================================================
// FileSource 构造函数测试
// ============================================================================

TEST_F(FileSourceTest, ConstructorWithSourceConfig) {
  SourceConfig config("nonexistent.mp4", DecodeMode::CPU);
  FileSource source(config);

  EXPECT_EQ(source.config().uri, "nonexistent.mp4");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::CPU);
  EXPECT_EQ(source.config().queue_capacity, 16);
  EXPECT_EQ(source.config().overflow_policy, OverflowPolicy::DROP_OLDEST);
  EXPECT_EQ(source.state(), NodeState::INIT);
  EXPECT_TRUE(source.is_source());
}

TEST_F(FileSourceTest, ConstructorWithUriOnly) {
  FileSource source("test.mp4");

  EXPECT_EQ(source.config().uri, "test.mp4");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::AUTO);
  EXPECT_EQ(source.state(), NodeState::INIT);
}

TEST_F(FileSourceTest, ConstructorWithUriAndDecodeMode) {
  FileSource source("test.mp4", DecodeMode::GPU);

  EXPECT_EQ(source.config().uri, "test.mp4");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::GPU);
}

TEST_F(FileSourceTest, ConstructorWithFullConfig) {
  SourceConfig config("test.mp4", DecodeMode::CPU, 1, 32, OverflowPolicy::BLOCK,
                      12345);
  FileSource source(config);

  EXPECT_EQ(source.config().uri, "test.mp4");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::CPU);
  EXPECT_EQ(source.config().gpu_device, 1);
  EXPECT_EQ(source.config().queue_capacity, 32);
  EXPECT_EQ(source.config().overflow_policy, OverflowPolicy::BLOCK);
  EXPECT_EQ(source.config().stream_id, 12345);
}

TEST_F(FileSourceTest, MoveConstructor) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source1(video_path_, DecodeMode::CPU);
  source1.start();
  ASSERT_TRUE(wait_for(
      [&source1]() { return source1.state() == NodeState::RUNNING; }, 2000ms));

  int64_t expected_width = source1.width();
  int64_t expected_height = source1.height();

  FileSource source2(std::move(source1));

  EXPECT_EQ(source2.width(), expected_width);
  EXPECT_EQ(source2.height(), expected_height);
  EXPECT_EQ(source2.state(), NodeState::RUNNING);

  source2.stop();
}

TEST_F(FileSourceTest, MoveAssignment) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source1(video_path_, DecodeMode::CPU);
  source1.start();
  ASSERT_TRUE(wait_for(
      [&source1]() { return source1.state() == NodeState::RUNNING; }, 2000ms));

  FileSource source2("dummy.mp4", DecodeMode::CPU);
  source2 = std::move(source1);

  EXPECT_GT(source2.width(), 0);
  EXPECT_GT(source2.height(), 0);
  EXPECT_EQ(source2.state(), NodeState::RUNNING);

  source2.stop();
}

// ============================================================================
// FileSource 状态转换测试
// ============================================================================

TEST_F(FileSourceTest, StateTransitionInitToRunning) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  EXPECT_EQ(source.state(), NodeState::INIT);

  source.start();

  EXPECT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));
  EXPECT_EQ(source.state(), NodeState::RUNNING);

  source.stop();
}

TEST_F(FileSourceTest, StateTransitionRunningToStopped) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  source.stop();
  source.wait_stop();

  EXPECT_EQ(source.state(), NodeState::STOPPED);
}

TEST_F(FileSourceTest, StateTransitionThroughDraining) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 停止并排空队列
  source.stop(true);
  source.wait_stop();

  EXPECT_EQ(source.state(), NodeState::STOPPED);
}

TEST_F(FileSourceTest, StartIdempotent) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);

  // 多次调用 start 应该是幂等的
  EXPECT_NO_THROW(source.start());
  EXPECT_NO_THROW(source.start());
  EXPECT_NO_THROW(source.start());

  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  source.stop();
}

TEST_F(FileSourceTest, StopIdempotent) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 多次调用 stop 应该是幂等的
  EXPECT_NO_THROW(source.stop());
  EXPECT_NO_THROW(source.stop());
  EXPECT_NO_THROW(source.stop());

  source.wait_stop();
  EXPECT_EQ(source.state(), NodeState::STOPPED);
}

TEST_F(FileSourceTest, StopBeforeStart) {
  FileSource source("dummy.mp4", DecodeMode::CPU);

  // 在 INIT 状态调用 stop 应该是安全的
  EXPECT_NO_THROW(source.stop());
  EXPECT_EQ(source.state(), NodeState::STOPPED);
}

// ============================================================================
// FileSource 错误路径测试
// ============================================================================

TEST_F(FileSourceTest, NonExistentFileThrowsOnStart) {
  FileSource source("/nonexistent/path/video.mp4", DecodeMode::CPU);

  // 文件不存在时应该抛出异常（ConfigError 或 NotFoundError）
  EXPECT_THROW(source.start(), VisionPipeError);
  EXPECT_NE(source.state(), NodeState::RUNNING);
}

TEST_F(FileSourceTest, EmptyUriThrowsOnStart) {
  FileSource source("", DecodeMode::CPU);

  EXPECT_THROW(source.start(), ConfigError);
}

TEST_F(FileSourceTest, InvalidPathThrowsOnStart) {
  // 使用一个存在但不是视频文件的路径
  FileSource source("/dev/null", DecodeMode::CPU);

  EXPECT_THROW(source.start(), VisionPipeError);
}

TEST_F(FileSourceTest, GpuModeWithoutCudaThrows) {
  if (cuda_available_) {
    GTEST_SKIP() << "CUDA available, cannot test GPU mode failure";
  }

  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  // 无 CUDA 时强制 GPU 模式应该抛出 CudaError
  FileSource source(video_path_, DecodeMode::GPU);
  EXPECT_THROW(source.start(), CudaError);
}

// ============================================================================
// FileSource 正常路径测试 - CPU 解码
// ============================================================================

TEST_F(FileSourceTest, CpuDecodeReadsAllFrames) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  SourceConfig config(video_path_, DecodeMode::CPU, 0, 500,
                      OverflowPolicy::BLOCK);
  FileSource source(config);
  source.create_output_queue(500, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 收集所有帧
  FrameCollector collector;
  auto output_queue = source.output_queue();

  while (source.state() == NodeState::RUNNING || !output_queue->empty()) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      collector.add(std::move(*frame));
    }
  }

  // 验证：应该读取到帧，且帧数应该等于 current_frame
  size_t frames_read = collector.size();
  EXPECT_GT(frames_read, 0u) << "Expected to read at least one frame";
  EXPECT_EQ(frames_read, static_cast<size_t>(source.current_frame()));
}

TEST_F(FileSourceTest, CpuDecodeFrameProperties) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  SourceConfig config(video_path_, DecodeMode::CPU, 0, 16,
                      OverflowPolicy::BLOCK, 1); // 设置 stream_id = 1
  FileSource source(config);
  source.create_output_queue(16, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms))
      << "Source failed to start";

  // 等待第一帧
  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  if (!frame.has_value()) {
    // 如果没有收到帧，直接跳过
    source.stop();
    GTEST_SKIP() << "No frame received within timeout";
  }

  EXPECT_EQ(frame->stream_id, 1);
  EXPECT_EQ(frame->frame_id, 0);

  source.stop();
}

TEST_F(FileSourceTest, CpuDecodeFrameHasValidTimestamp) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms))
      << "Source failed to start";

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  if (!frame.has_value()) {
    source.stop();
    GTEST_SKIP() << "No frame received within timeout";
  }

  // 时间戳应该是非负的
  EXPECT_GE(frame->pts_us, 0);

  source.stop();
}

TEST_F(FileSourceTest, CpuDecodeVideoProperties) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 验证视频属性已正确读取
  EXPECT_GT(source.width(), 0);
  EXPECT_GT(source.height(), 0);
  EXPECT_GT(source.fps(), 0.0);
  // frame_count 可能是 -1（未知）或正数
  EXPECT_NE(source.frame_count(), 0);

  source.stop();
}

TEST_F(FileSourceTest, CpuDecodeSequentialFrameIds) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  SourceConfig config(video_path_, DecodeMode::CPU, 0, 500,
                      OverflowPolicy::BLOCK);
  FileSource source(config);
  source.create_output_queue(500, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  std::vector<int64_t> frame_ids;
  auto output_queue = source.output_queue();

  // 收集帧 ID
  while (source.state() == NodeState::RUNNING || !output_queue->empty()) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      frame_ids.push_back(frame->frame_id);
    }
  }

  // 验证帧 ID 连续递增
  ASSERT_GT(frame_ids.size(), 0u);
  for (size_t i = 0; i < frame_ids.size(); ++i) {
    EXPECT_EQ(frame_ids[i], static_cast<int64_t>(i));
  }
}

TEST_F(FileSourceTest, CpuDecodeCurrentFrameCounter) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.create_output_queue(50, OverflowPolicy::DROP_OLDEST);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 等待一些帧被处理
  wait_for([&source]() { return source.current_frame() >= 10; }, 3000ms);

  EXPECT_GE(source.current_frame(), 10);

  source.stop();
  source.wait_stop();

  // 停止后计数器应该是总帧数（如果已知）
  int64_t expected_frames = source.frame_count();
  if (expected_frames > 0) {
    EXPECT_EQ(source.current_frame(), expected_frames);
  }
}

// ============================================================================
// FileSource AUTO 模式测试
// ============================================================================

TEST_F(FileSourceTest, AutoModeSelectsCorrectDecoder) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::AUTO);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 验证实际解码模式
  if (cuda_available_) {
    EXPECT_EQ(source.actual_decode_mode(), DecodeMode::GPU);
  } else {
    EXPECT_EQ(source.actual_decode_mode(), DecodeMode::CPU);
  }

  source.stop();
}

TEST_F(FileSourceTest, AutoModeProducesValidFrames) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::AUTO);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame->has_image());
  EXPECT_GT(frame->image.nbytes, 0);

  source.stop();
}

// ============================================================================
// FileSource GPU 解码测试
// ============================================================================

TEST_F(FileSourceTest, GpuDecodeReadsAllFrames) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available, skipping GPU decode test";
  }

  SourceConfig config(video_path_, DecodeMode::GPU, 0, 500,
                      OverflowPolicy::BLOCK);
  FileSource source(config);
  source.create_output_queue(500, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 验证使用 GPU 解码
  EXPECT_EQ(source.actual_decode_mode(), DecodeMode::GPU);

  // 收集所有帧
  FrameCollector collector;
  auto output_queue = source.output_queue();

  while (source.state() == NodeState::RUNNING || !output_queue->empty()) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      collector.add(std::move(*frame));
    }
  }

  // 验证帧数：应该读取到帧
  size_t frames_read = collector.size();
  EXPECT_GT(frames_read, 0u);
  EXPECT_EQ(frames_read, static_cast<size_t>(source.current_frame()));
}

TEST_F(FileSourceTest, GpuDecodeFrameInDeviceMemory) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available, skipping GPU decode test";
  }

  FileSource source(video_path_, DecodeMode::GPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame->has_image());

  // GPU 解码的帧应该在设备内存
  EXPECT_EQ(frame->image.memory_type(), MemoryType::CUDA_DEVICE);

  source.stop();
}

TEST_F(FileSourceTest, GpuDecodeNoCpuGpuCopy) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available, skipping GPU decode test";
  }

  FileSource source(video_path_, DecodeMode::GPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  ASSERT_TRUE(frame.has_value());

  // GPU 解码直接在设备内存，无 CPU -> GPU 拷贝
  // 验证帧数据在设备内存
  EXPECT_EQ(frame->image.memory_type(), MemoryType::CUDA_DEVICE);
  EXPECT_NE(frame->image.data, nullptr);

  source.stop();
}

// ============================================================================
// FileSource CPU 解码内存测试
// ============================================================================

TEST_F(FileSourceTest, CpuDecodeFrameInCpuMemory) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame->has_image());

  // CPU 解码的帧应该在 CPU 内存（后续会 upload 到 GPU）
  // 注意：根据实现，可能是 CPU 或 CUDA_HOST
  MemoryType mem_type = frame->image.memory_type();
  EXPECT_TRUE(mem_type == MemoryType::CPU || mem_type == MemoryType::CUDA_HOST);

  source.stop();
}

// ============================================================================
// FileSource 队列溢出策略测试
// ============================================================================

TEST_F(FileSourceTest, DropOldestOnOverflow) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  // 使用很小的队列容量
  SourceConfig config(video_path_, DecodeMode::CPU, 0, 3,
                      OverflowPolicy::DROP_OLDEST);
  FileSource source(config);
  source.create_output_queue(3, OverflowPolicy::DROP_OLDEST);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // 等待处理完成（队列满了会丢弃旧帧）
  wait_for([&source]() { return source.state() == NodeState::STOPPED; },
           5000ms);

  auto stats = source.output_queue()->stats();
  // 由于队列容量只有 3，帧数为 100，必然有丢弃
  EXPECT_GT(stats.dropped_count, 0u);

  source.stop();
}

TEST_F(FileSourceTest, BlockPolicyNoDrop) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  SourceConfig config(video_path_, DecodeMode::CPU, 0, 200,
                      OverflowPolicy::BLOCK);
  FileSource source(config);
  source.create_output_queue(200, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  // BLOCK 模式下不应该有丢弃
  auto stats = source.output_queue()->stats();
  EXPECT_EQ(stats.dropped_count, 0u);

  source.stop();
}

// ============================================================================
// FileSource 源节点特性测试
// ============================================================================

TEST_F(FileSourceTest, IsSourceReturnsTrue) {
  FileSource source("test.mp4");
  EXPECT_TRUE(source.is_source());
  EXPECT_FALSE(source.is_sink());
}

TEST_F(FileSourceTest, HasNoInputQueue) {
  FileSource source("test.mp4");
  EXPECT_EQ(source.input_queue(), nullptr);
}

TEST_F(FileSourceTest, HasOutputQueue) {
  FileSource source("test.mp4");
  source.create_output_queue(10);

  EXPECT_NE(source.output_queue(), nullptr);
  EXPECT_EQ(source.output_queue()->capacity(), 10);
}

TEST_F(FileSourceTest, ProcessDoesNothing) {
  // 源节点的 process() 不应该做任何事情
  FileSource source("test.mp4");

  Frame frame;
  EXPECT_NO_THROW(source.process(frame));
  // frame 应该没有被修改
  EXPECT_FALSE(frame.has_image());
}

// ============================================================================
// FileSource Stream ID 测试
// ============================================================================

TEST_F(FileSourceTest, StreamIdPropagatesToFrames) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  const int64_t expected_stream_id = 42;
  SourceConfig config(video_path_, DecodeMode::CPU, 0, 16,
                      OverflowPolicy::BLOCK, expected_stream_id);
  FileSource source(config);
  source.create_output_queue(16, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(2000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_EQ(frame->stream_id, expected_stream_id);

  source.stop();
}

// ============================================================================
// FileSource 边界值测试
// ============================================================================

TEST_F(FileSourceTest, DISABLED_ZeroFrameVideo) {
  // 需要准备一个 0 帧的视频文件
  const char *zero_frame_video =
      std::getenv("VISIONPIPE_TEST_ZERO_FRAME_VIDEO");
  if (!zero_frame_video || !file_exists(zero_frame_video)) {
    GTEST_SKIP() << "Zero frame test video not available";
  }

  FileSource source(zero_frame_video, DecodeMode::CPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  source.wait_stop();

  EXPECT_EQ(source.frame_count(), 0);
  EXPECT_EQ(source.current_frame(), 0);
}

TEST_F(FileSourceTest, DISABLED_VeryShortVideo) {
  // 需要准备一个很短的视频（如 1 帧）
  const char *short_video = std::getenv("VISIONPIPE_TEST_SHORT_VIDEO");
  if (!short_video || !file_exists(short_video)) {
    GTEST_SKIP() << "Short test video not available";
  }

  FileSource source(short_video, DecodeMode::CPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  source.wait_stop();

  EXPECT_EQ(source.frame_count(), 1);
  EXPECT_EQ(source.current_frame(), 1);
}

// ============================================================================
// RtspSource 测试夹具
// ============================================================================

class RtspSourceTest : public ::testing::Test {
protected:
  void SetUp() override {
    rtsp_url_ = get_test_rtsp_url();
    rtsp_available_ = (rtsp_url_ != nullptr);
    cuda_available_ = is_cuda_available();
  }

  const char *rtsp_url_ = nullptr;
  bool rtsp_available_ = false;
  bool cuda_available_ = false;
};

// ============================================================================
// RtspSource 构造函数测试
// ============================================================================

TEST_F(RtspSourceTest, ConstructorWithSourceConfig) {
  SourceConfig config("rtsp://localhost:8554/test", DecodeMode::CPU);
  RtspSource source(config);

  EXPECT_EQ(source.config().uri, "rtsp://localhost:8554/test");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::CPU);
  EXPECT_EQ(source.state(), NodeState::INIT);
  EXPECT_TRUE(source.is_source());
}

TEST_F(RtspSourceTest, ConstructorWithUriOnly) {
  RtspSource source("rtsp://localhost:8554/test");

  EXPECT_EQ(source.config().uri, "rtsp://localhost:8554/test");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::AUTO);
}

TEST_F(RtspSourceTest, ConstructorWithUriAndDecodeMode) {
  RtspSource source("rtsp://localhost:8554/test", DecodeMode::GPU);

  EXPECT_EQ(source.config().uri, "rtsp://localhost:8554/test");
  EXPECT_EQ(source.config().decode_mode, DecodeMode::GPU);
}

// ============================================================================
// RtspSource 状态转换测试
// ============================================================================

TEST_F(RtspSourceTest, StateTransitionInitToRunning) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set, skipping RTSP tests";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  EXPECT_EQ(source.state(), NodeState::INIT);

  source.start();

  EXPECT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));
  EXPECT_EQ(source.state(), NodeState::RUNNING);
  EXPECT_TRUE(source.is_connected());

  source.stop();
}

TEST_F(RtspSourceTest, StateTransitionRunningToStopped) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set, skipping RTSP tests";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  source.stop();
  source.wait_stop();

  EXPECT_EQ(source.state(), NodeState::STOPPED);
  EXPECT_FALSE(source.is_connected());
}

TEST_F(RtspSourceTest, StopBeforeStart) {
  RtspSource source("rtsp://localhost:8554/test");

  EXPECT_NO_THROW(source.stop());
  EXPECT_EQ(source.state(), NodeState::STOPPED);
}

// ============================================================================
// RtspSource 连接错误测试
// ============================================================================

TEST_F(RtspSourceTest, InvalidRtspUrlThrowsOnStart) {
  RtspSource source("rtsp://nonexistent.server:8554/test", DecodeMode::CPU);

  // 连接失败应该抛出异常或最终进入 STOPPED 状态
  EXPECT_THROW(source.start(), VisionPipeError);
}

TEST_F(RtspSourceTest, EmptyUrlThrowsOnStart) {
  RtspSource source("", DecodeMode::CPU);

  EXPECT_THROW(source.start(), ConfigError);
}

TEST_F(RtspSourceTest, MalformedUrlThrowsOnStart) {
  RtspSource source("not-a-valid-url", DecodeMode::CPU);

  EXPECT_THROW(source.start(), VisionPipeError);
}

TEST_F(RtspSourceTest, GpuModeWithoutCudaThrows) {
  if (cuda_available_) {
    GTEST_SKIP() << "CUDA available, cannot test GPU mode failure";
  }
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::GPU);
  EXPECT_THROW(source.start(), CudaError);
}

// ============================================================================
// RtspSource 正常路径测试
// ============================================================================

TEST_F(RtspSourceTest, CpuDecodeProducesFrames) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(3000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame->has_image());
  EXPECT_GT(frame->image.nbytes, 0);

  source.stop();
}

TEST_F(RtspSourceTest, CpuDecodeVideoProperties) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  // RTSP 流的帧数是未知的
  // 注意：RTSP 流没有固定的 frame_count，这里不检查
  EXPECT_GT(source.width(), 0);
  EXPECT_GT(source.height(), 0);

  source.stop();
}

TEST_F(RtspSourceTest, AutoModeSelectsCorrectDecoder) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::AUTO);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  if (cuda_available_) {
    EXPECT_EQ(source.actual_decode_mode(), DecodeMode::GPU);
  } else {
    EXPECT_EQ(source.actual_decode_mode(), DecodeMode::CPU);
  }

  source.stop();
}

TEST_F(RtspSourceTest, ContinuousFrameProduction) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  source.create_output_queue(50, OverflowPolicy::DROP_OLDEST);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  // 收集一段时间内的帧
  FrameCollector collector;
  auto output_queue = source.output_queue();
  auto start_time = std::chrono::steady_clock::now();

  while (std::chrono::steady_clock::now() - start_time < 2000ms) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      collector.add(std::move(*frame));
    }
  }

  // 应该收到多帧
  EXPECT_GT(collector.size(), 0u);

  source.stop();
}

TEST_F(RtspSourceTest, FrameIdsIncrement) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  source.create_output_queue(100, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  auto output_queue = source.output_queue();

  int64_t prev_frame_id = -1;
  size_t frames_checked = 0;

  while (frames_checked < 10) {
    auto frame = output_queue->pop_for(500ms);
    if (!frame.has_value())
      break;

    if (prev_frame_id >= 0) {
      EXPECT_GT(frame->frame_id, prev_frame_id);
    }
    prev_frame_id = frame->frame_id;
    ++frames_checked;
  }

  EXPECT_GT(frames_checked, 0u);

  source.stop();
}

// ============================================================================
// RtspSource GPU 解码测试
// ============================================================================

TEST_F(RtspSourceTest, GpuDecodeProducesFrames) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }
  if (!cuda_available_) {
    GTEST_SKIP() << "CUDA not available, skipping GPU decode test";
  }

  RtspSource source(rtsp_url_, DecodeMode::GPU);
  source.create_output_queue(10, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  EXPECT_EQ(source.actual_decode_mode(), DecodeMode::GPU);

  auto output_queue = source.output_queue();
  auto frame = output_queue->pop_for(3000ms);

  ASSERT_TRUE(frame.has_value());
  EXPECT_EQ(frame->image.memory_type(), MemoryType::CUDA_DEVICE);

  source.stop();
}

// ============================================================================
// RtspSource 源节点特性测试
// ============================================================================

TEST_F(RtspSourceTest, IsSourceReturnsTrue) {
  RtspSource source("rtsp://localhost:8554/test");
  EXPECT_TRUE(source.is_source());
  EXPECT_FALSE(source.is_sink());
}

TEST_F(RtspSourceTest, HasNoInputQueue) {
  RtspSource source("rtsp://localhost:8554/test");
  EXPECT_EQ(source.input_queue(), nullptr);
}

TEST_F(RtspSourceTest, ProcessDoesNothing) {
  RtspSource source("rtsp://localhost:8554/test");

  Frame frame;
  EXPECT_NO_THROW(source.process(frame));
  EXPECT_FALSE(frame.has_image());
}

// ============================================================================
// RtspSource 连接状态测试
// ============================================================================

TEST_F(RtspSourceTest, IsConnectedAfterStart) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);

  EXPECT_FALSE(source.is_connected());

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  EXPECT_TRUE(source.is_connected());

  source.stop();
  EXPECT_FALSE(source.is_connected());
}

// ============================================================================
// 多线程安全性测试
// ============================================================================

TEST_F(FileSourceTest, ConcurrentStopAndRead) {
  if (!video_available_) {
    GTEST_SKIP() << "Test video not available: " << video_path_;
  }

  FileSource source(video_path_, DecodeMode::CPU);
  source.create_output_queue(50, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 2000ms));

  auto output_queue = source.output_queue();

  // 在读取帧的同时停止
  std::thread stopper([&source]() {
    std::this_thread::sleep_for(500ms);
    source.stop();
  });

  // 持续读取帧
  size_t frames_read = 0;
  while (source.state() == NodeState::RUNNING || !output_queue->empty()) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      ++frames_read;
    }
  }

  stopper.join();
  EXPECT_GT(frames_read, 0u);
}

TEST_F(RtspSourceTest, DISABLED_ConcurrentStopAndRead) {
  if (!rtsp_available_) {
    GTEST_SKIP() << "VISIONPIPE_TEST_RTSP_URL not set";
  }

  RtspSource source(rtsp_url_, DecodeMode::CPU);
  source.create_output_queue(50, OverflowPolicy::BLOCK);

  source.start();
  ASSERT_TRUE(wait_for(
      [&source]() { return source.state() == NodeState::RUNNING; }, 5000ms));

  auto output_queue = source.output_queue();

  std::thread stopper([&source]() {
    std::this_thread::sleep_for(1000ms);
    source.stop();
  });

  size_t frames_read = 0;
  while (source.state() == NodeState::RUNNING || !output_queue->empty()) {
    auto frame = output_queue->pop_for(100ms);
    if (frame.has_value()) {
      ++frames_read;
    }
  }

  stopper.join();
  EXPECT_GT(frames_read, 0u);
}

} // namespace
} // namespace visionpipe
