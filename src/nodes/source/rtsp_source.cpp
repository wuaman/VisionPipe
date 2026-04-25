#include "nodes/source/rtsp_source.h"

#include <chrono>
#include <thread>

#include "core/logger.h"

#include <opencv2/videoio.hpp>

namespace visionpipe {

RtspSource::RtspSource(const SourceConfig &config)
    : NodeBase("RtspSource:" + config.uri), config_(config),
      actual_decode_mode_(config.decode_mode) {
  create_output_queue(config_.queue_capacity, config_.overflow_policy);
}

RtspSource::RtspSource(const std::string &uri, DecodeMode mode)
    : RtspSource(SourceConfig(uri, mode)) {}

RtspSource::~RtspSource() {
  stop(false);
  if (source_thread_.joinable()) {
    source_thread_.join();
  }
}

RtspSource::RtspSource(RtspSource &&other) noexcept
    : NodeBase(std::move(other)), config_(std::move(other.config_)),
      actual_decode_mode_(other.actual_decode_mode_),
      capture_(std::move(other.capture_)), width_(other.width_),
      height_(other.height_), fps_(other.fps_),
      current_frame_(other.current_frame_.load()),
      connected_(other.connected_.load()) {
  other.width_ = 0;
  other.height_ = 0;
  other.fps_ = 0.0;
}

RtspSource &RtspSource::operator=(RtspSource &&other) noexcept {
  if (this != &other) {
    stop(false);
    if (source_thread_.joinable()) {
      source_thread_.join();
    }

    NodeBase::operator=(std::move(other));
    config_ = std::move(other.config_);
    actual_decode_mode_ = other.actual_decode_mode_;
    capture_ = std::move(other.capture_);
    width_ = other.width_;
    height_ = other.height_;
    fps_ = other.fps_;
    current_frame_ = other.current_frame_.load();
    connected_ = other.connected_.load();

    other.width_ = 0;
    other.height_ = 0;
    other.fps_ = 0.0;
  }
  return *this;
}

void RtspSource::process(Frame &frame) {
  // 源节点不使用 process() 接口
  (void)frame;
}

void RtspSource::start() {
  if (state_ == NodeState::RUNNING) {
    return;
  }

  on_init();
  state_ = NodeState::RUNNING;

  // 启动专用线程从 RTSP 流读取帧
  source_thread_ = std::thread(&RtspSource::source_worker_loop, this);

  VP_LOG_INFO(
      "RtspSource '{}' started, decode_mode={}, resolution={}x{}, fps={}",
      name_, static_cast<int>(actual_decode_mode_), width_, height_, fps_);
}

void RtspSource::stop(bool drain) {
  NodeBase::stop(drain);
  if (output_queue_) {
    output_queue_->stop();
  }
}

void RtspSource::on_init() {
  // RTSP 流目前统一使用 CPU 解码（OpenCV 对 RTSP + CUDA 支持有限）
  // TODO: 后续可探索 FFmpeg + NVCUVID 方案

  if (config_.decode_mode == DecodeMode::GPU) {
    VP_LOG_WARN(
        "RtspSource '{}' GPU decode mode requested but falling back to CPU "
        "(RTSP GPU decode not fully supported yet)",
        name_);
  }

  init_cpu_decoder();
  actual_decode_mode_ = DecodeMode::CPU;
}

void RtspSource::on_stop() {
  connected_ = false;
  capture_.reset();
}

void RtspSource::source_worker_loop() {
  VP_LOG_DEBUG("RtspSource '{}' worker thread started", name_);

  Frame frame;
  frame.stream_id = config_.stream_id;

  while (state_ == NodeState::RUNNING) {
    if (!connected_ && !try_reconnect()) {
      // 重连失败，等待后重试
      std::this_thread::sleep_for(
          std::chrono::milliseconds(reconnect_delay_ms));
      continue;
    }

    bool read_success = read_frame(frame);

    if (!read_success) {
      VP_LOG_WARN("RtspSource '{}' read failed, attempting reconnect", name_);
      connected_ = false;
      continue;
    }

    frame.frame_id = current_frame_.fetch_add(1);
    ++processed_count_;

    // 推送到输出队列
    if (output_queue_) {
      output_queue_->push(std::move(frame));
    }

    // 重置 frame 以复用
    frame = Frame();
    frame.stream_id = config_.stream_id;
  }

  state_ = NodeState::STOPPED;
  VP_LOG_INFO("RtspSource '{}' stopped, total frames: {}", name_,
              current_frame_.load());
}

bool RtspSource::try_init_gpu_decoder() {
  // RTSP 流的 GPU 解码目前不支持
  // 需要使用 FFmpeg + NVCUVID 或更复杂的方案
  VP_LOG_DEBUG("RTSP GPU decode not implemented, using CPU fallback");
  return false;
}

void RtspSource::init_cpu_decoder() {
  capture_ = std::make_unique<cv::VideoCapture>();

  // 设置 RTSP 专用参数
  capture_->set(cv::CAP_PROP_BUFFERSIZE, 1); // 最小化缓冲以降低延迟

  // 打开 RTSP 流
  if (!capture_->open(config_.uri, cv::CAP_FFMPEG)) {
    throw ConfigError("Failed to open RTSP stream: " + config_.uri);
  }

  // 获取视频属性
  width_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
  fps_ = capture_->get(cv::CAP_PROP_FPS);
  if (fps_ <= 0) {
    fps_ = 25.0; // 默认值
  }

  connected_ = true;

  VP_LOG_INFO("RtspSource '{}' CPU decoder initialized: {}x{}, fps={}", name_,
              width_, height_, fps_);
}

bool RtspSource::read_frame(Frame &frame) {
  if (!capture_ || !capture_->isOpened()) {
    return false;
  }

  try {
    cv::Mat cpu_frame;
    if (!capture_->read(cpu_frame)) {
      return false;
    }

    // 获取帧时间戳
    frame.pts_us =
        static_cast<int64_t>(capture_->get(cv::CAP_PROP_POS_MSEC) * 1000);

    // TODO: 实现从 cv::Mat 到 Tensor 的转换

    return true;
  } catch (const cv::Exception &e) {
    VP_LOG_ERROR("RtspSource '{}' read failed: {}", name_, e.what());
    return false;
  }
}

bool RtspSource::try_reconnect() {
  for (int attempt = 1; attempt <= max_reconnect_attempts; ++attempt) {
    VP_LOG_INFO("RtspSource '{}' reconnect attempt {}/{}", name_, attempt,
                max_reconnect_attempts);

    capture_.reset();
    capture_ = std::make_unique<cv::VideoCapture>();
    capture_->set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (capture_->open(config_.uri, cv::CAP_FFMPEG)) {
      connected_ = true;
      VP_LOG_INFO("RtspSource '{}' reconnected successfully", name_);
      return true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
  }

  VP_LOG_ERROR("RtspSource '{}' failed to reconnect after {} attempts", name_,
               max_reconnect_attempts);
  return false;
}

} // namespace visionpipe
