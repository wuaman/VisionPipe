#include "nodes/source/file_source.h"

#include <cstring>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "core/error.h"
#include "core/logger.h"
#include "core/tensor.h"

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace visionpipe {

FileSource::FileSource(const SourceConfig &config)
    : SourceNode("FileSource:" + config.uri, config),
      actual_decode_mode_(config.decode_mode) {}

FileSource::FileSource(const std::string &uri, DecodeMode mode)
    : FileSource(SourceConfig(uri, mode)) {}

FileSource::FileSource(FileSource &&other) noexcept
    : SourceNode(std::move(other)),
      actual_decode_mode_(other.actual_decode_mode_),
      cpu_capture_(std::move(other.cpu_capture_))
#ifdef VISIONPIPE_USE_CUDA
      ,
      gpu_reader_(other.gpu_reader_)
#endif
      ,
      width_(other.width_), height_(other.height_), fps_(other.fps_),
      frame_count_(other.frame_count_), current_frame_(other.current_frame_) {
#ifdef VISIONPIPE_USE_CUDA
  other.gpu_reader_ = nullptr;
#endif
  other.width_ = 0;
  other.height_ = 0;
  other.fps_ = 0.0;
  other.frame_count_ = -1;
  other.current_frame_ = 0;
}

FileSource &FileSource::operator=(FileSource &&other) noexcept {
  if (this != &other) {
    SourceNode::operator=(std::move(other));
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
    current_frame_ = other.current_frame_;

    other.width_ = 0;
    other.height_ = 0;
    other.fps_ = 0.0;
    other.frame_count_ = -1;
    other.current_frame_ = 0;
  }
  return *this;
}

void FileSource::on_open() {
  if (config_.uri.empty()) {
    throw ConfigError("Video file URI is empty");
  }

  bool gpu_init_success = false;
  current_frame_ = 0;

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

  VP_LOG_INFO(
      "FileSource '{}' opened, decode_mode={}, resolution={}x{}, fps={}",
      name_, static_cast<int>(actual_decode_mode_), width_, height_, fps_);
}

bool FileSource::read_next(Frame &frame) {
  bool read_success = false;

  if (actual_decode_mode_ == DecodeMode::GPU) {
    read_success = read_frame_gpu(frame);
  } else {
    read_success = read_frame_cpu(frame);
  }

  if (read_success) {
    ++current_frame_;
  }

  return read_success;
}

void FileSource::on_close() {
  cpu_capture_.reset();
#ifdef VISIONPIPE_USE_CUDA
  gpu_reader_ = nullptr;
#endif
}

bool FileSource::try_init_gpu_decoder() {
#ifdef VISIONPIPE_USE_CUDA
  if (!is_nvdec_available()) {
    VP_LOG_DEBUG("NVCUVID not available on this system");
    return false;
  }

  try {
    gpu_reader_ = cv::cudacodec::createVideoReader(config_.uri);

    cv::cuda::GpuMat first_frame;
    if (!gpu_reader_->nextFrame(first_frame)) {
      VP_LOG_WARN("FileSource '{}' failed to read first frame for GPU init",
                  name_);
      gpu_reader_ = nullptr;
      return false;
    }

    width_ = first_frame.cols;
    height_ = first_frame.rows;

    gpu_reader_ = cv::cudacodec::createVideoReader(config_.uri);

    fps_ = 25.0;

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

    frame.pts_us =
        current_frame_ * static_cast<int64_t>(1e6 / (fps_ > 0 ? fps_ : 25.0));

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

    cv::Mat rgb;
    cv::cvtColor(cpu_frame, rgb, cv::COLOR_BGR2RGB);

    static CpuAllocator cpu_alloc;
    const int h = rgb.rows, w = rgb.cols, c = rgb.channels();
    frame.image = Tensor({static_cast<int64_t>(h),
                          static_cast<int64_t>(w),
                          static_cast<int64_t>(c)},
                         DataType::UINT8, &cpu_alloc);
    std::memcpy(frame.image.data, rgb.data, frame.image.nbytes);

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
