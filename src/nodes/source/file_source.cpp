#include "nodes/source/file_source.h"

#include <chrono>

#include "core/logger.h"

#include <opencv2/videoio.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace visionpipe {

FileSource::FileSource(const SourceConfig &config)
    : NodeBase("FileSource:" + config.uri), config_(config),
      actual_decode_mode_(config.decode_mode) {
  create_output_queue(config_.queue_capacity, config_.overflow_policy);
}

FileSource::FileSource(const std::string &uri, DecodeMode mode)
    : FileSource(SourceConfig(uri, mode)) {}

FileSource::~FileSource() {
  stop(false);
  if (source_thread_.joinable()) {
    source_thread_.join();
  }
}

FileSource::FileSource(FileSource &&other) noexcept
    : NodeBase(std::move(other)), config_(std::move(other.config_)),
      actual_decode_mode_(other.actual_decode_mode_),
      cpu_capture_(std::move(other.cpu_capture_))
#ifdef VISIONPIPE_USE_CUDA
      ,
      gpu_reader_(other.gpu_reader_)
#endif
      ,
      width_(other.width_), height_(other.height_), fps_(other.fps_),
      frame_count_(other.frame_count_),
      current_frame_(other.current_frame_.load()) {
#ifdef VISIONPIPE_USE_CUDA
  other.gpu_reader_ = nullptr;
#endif
  other.width_ = 0;
  other.height_ = 0;
  other.fps_ = 0.0;
  other.frame_count_ = -1;
}

FileSource &FileSource::operator=(FileSource &&other) noexcept {
  if (this != &other) {
    stop(false);
    if (source_thread_.joinable()) {
      source_thread_.join();
    }

    NodeBase::operator=(std::move(other));
    config_ = std::move(other.config_);
    actual_decode_mode_ = other.actual_decode_mode_;
    cpu_capture_ = std::move(other.cpu_capture_);
#ifdef VISIONPIPE_USE_CUDA
    gpu_reader_ = other.gpu_reader_;
    other.gpu_reader_ = nullptr;
#endif
    width_ = other.width_;
    height_ = other.height_;
    fps_ = other.fps_;
    frame_count_ = other.frame_count_;
    current_frame_ = other.current_frame_.load();

    other.width_ = 0;
    other.height_ = 0;
    other.fps_ = 0.0;
    other.frame_count_ = -1;
  }
  return *this;
}

void FileSource::process(Frame &frame) { (void)frame; }

void FileSource::start() {
  if (state_ == NodeState::RUNNING) {
    return;
  }

  on_init();
  state_ = NodeState::RUNNING;

  source_thread_ = std::thread(&FileSource::source_worker_loop, this);

  VP_LOG_INFO(
      "FileSource '{}' started, decode_mode={}, resolution={}x{}, fps={}",
      name_, static_cast<int>(actual_decode_mode_), width_, height_, fps_);
}

void FileSource::stop(bool drain) {
  NodeBase::stop(drain);
  if (output_queue_) {
    output_queue_->stop();
  }
}

void FileSource::on_init() {
  bool gpu_init_success = false;

  switch (config_.decode_mode) {
  case DecodeMode::AUTO:
    gpu_init_success = try_init_gpu_decoder();
    if (!gpu_init_success) {
      VP_LOG_INFO(
          "FileSource '{}' NVCUVID not available, falling back to CPU decode",
          name_);
      init_cpu_decoder();
      actual_decode_mode_ = DecodeMode::CPU;
    } else {
      actual_decode_mode_ = DecodeMode::GPU;
    }
    break;

  case DecodeMode::GPU:
    gpu_init_success = try_init_gpu_decoder();
    if (!gpu_init_success) {
      throw CudaError("GPU decode requested but NVCUVID not available for: " +
                      config_.uri);
    }
    actual_decode_mode_ = DecodeMode::GPU;
    break;

  case DecodeMode::CPU:
    init_cpu_decoder();
    actual_decode_mode_ = DecodeMode::CPU;
    break;
  }
}

void FileSource::on_stop() {
  cpu_capture_.reset();
#ifdef VISIONPIPE_USE_CUDA
  gpu_reader_ = nullptr;
#endif
}

void FileSource::source_worker_loop() {
  VP_LOG_DEBUG("FileSource '{}' worker thread started", name_);

  Frame frame;
  frame.stream_id = config_.stream_id;

  while (state_ == NodeState::RUNNING) {
    bool read_success = false;

    if (actual_decode_mode_ == DecodeMode::GPU) {
      read_success = read_frame_gpu(frame);
    } else {
      read_success = read_frame_cpu(frame);
    }

    if (!read_success) {
      VP_LOG_INFO(
          "FileSource '{}' reached end of video or read failed at frame {}",
          name_, current_frame_.load());
      break;
    }

    frame.frame_id = current_frame_.fetch_add(1);
    ++processed_count_;

    if (output_queue_) {
      output_queue_->push(std::move(frame));
    }

    frame = Frame();
    frame.stream_id = config_.stream_id;
  }

  state_ = NodeState::STOPPED;
  VP_LOG_INFO("FileSource '{}' stopped, total frames: {}", name_,
              current_frame_.load());
}

bool FileSource::try_init_gpu_decoder() {
#ifdef VISIONPIPE_USE_CUDA
  if (!is_nvdec_available()) {
    VP_LOG_DEBUG("NVCUVID not available on this system");
    return false;
  }

  try {
    // 使用 createVideoReader 工厂函数创建 GPU 视频阅读器
    gpu_reader_ = cv::cudacodec::createVideoReader(config_.uri);

    // 读取第一帧以获取实际分辨率
    cv::cuda::GpuMat first_frame;
    if (!gpu_reader_->nextFrame(first_frame)) {
      VP_LOG_WARN("FileSource '{}' failed to read first frame for GPU init",
                  name_);
      gpu_reader_ = nullptr;
      return false;
    }

    width_ = first_frame.cols;
    height_ = first_frame.rows;

    // 重新创建以从头播放
    gpu_reader_ = cv::cudacodec::createVideoReader(config_.uri);

    fps_ = 25.0; // 默认值，cv::cudacodec 不直接提供帧率

    VP_LOG_INFO("FileSource '{}' GPU decoder initialized: {}x{}", name_, width_,
                height_);
    return true;

  } catch (const cv::Exception &e) {
    VP_LOG_WARN("FileSource '{}' GPU decoder init failed: {}", name_, e.what());
    gpu_reader_ = nullptr;
    return false;
  } catch (const std::exception &e) {
    VP_LOG_WARN("FileSource '{}' GPU decoder init failed: {}", name_, e.what());
    gpu_reader_ = nullptr;
    return false;
  }
#else
  VP_LOG_DEBUG("VisionPipe compiled without CUDA support");
  return false;
#endif
}

void FileSource::init_cpu_decoder() {
  cpu_capture_ = std::make_unique<cv::VideoCapture>(config_.uri);

  if (!cpu_capture_->isOpened()) {
    throw ConfigError("Failed to open video file: " + config_.uri);
  }

  width_ = static_cast<int>(cpu_capture_->get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int>(cpu_capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
  fps_ = cpu_capture_->get(cv::CAP_PROP_FPS);
  frame_count_ =
      static_cast<int64_t>(cpu_capture_->get(cv::CAP_PROP_FRAME_COUNT));

  VP_LOG_INFO(
      "FileSource '{}' CPU decoder initialized: {}x{}, fps={}, frames={}",
      name_, width_, height_, fps_, frame_count_);
}

bool FileSource::read_frame_gpu(Frame &frame) {
#ifdef VISIONPIPE_USE_CUDA
  if (!gpu_reader_) {
    return false;
  }

  try {
    cv::cuda::GpuMat gpu_frame;
    if (!gpu_reader_->nextFrame(gpu_frame)) {
      return false;
    }

    // GPU 解码帧直接在显存中
    frame.pts_us =
        current_frame_ * static_cast<int64_t>(1e6 / (fps_ > 0 ? fps_ : 25.0));

    // TODO: 实现从 GpuMat 到 Tensor 的零拷贝包装

    return true;
  } catch (const cv::Exception &e) {
    VP_LOG_ERROR("FileSource '{}' GPU read failed: {}", name_, e.what());
    return false;
  }
#else
  (void)frame;
  return false;
#endif
}

bool FileSource::read_frame_cpu(Frame &frame) {
  if (!cpu_capture_ || !cpu_capture_->isOpened()) {
    return false;
  }

  try {
    cv::Mat cpu_frame;
    if (!cpu_capture_->read(cpu_frame)) {
      return false;
    }

    frame.pts_us =
        static_cast<int64_t>(cpu_capture_->get(cv::CAP_PROP_POS_MSEC) * 1000);

    // TODO: 实现从 cv::Mat 到 Tensor 的转换

    return true;
  } catch (const cv::Exception &e) {
    VP_LOG_ERROR("FileSource '{}' CPU read failed: {}", name_, e.what());
    return false;
  }
}

bool FileSource::is_nvdec_available() {
#ifdef VISIONPIPE_USE_CUDA
  try {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

} // namespace visionpipe
