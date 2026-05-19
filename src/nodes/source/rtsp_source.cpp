#include "nodes/source/rtsp_source.h"

#include <opencv2/videoio.hpp>

#include "core/error.h"
#include "core/logger.h"

namespace visionpipe {

RtspSource::RtspSource(const SourceConfig &config)
    : SourceNode("RtspSource:" + config.uri, config),
      actual_decode_mode_(config.decode_mode) {}

RtspSource::RtspSource(const std::string &uri, DecodeMode mode)
    : RtspSource(SourceConfig(uri, mode)) {}

RtspSource::RtspSource(RtspSource &&other) noexcept
    : SourceNode(std::move(other)),
      actual_decode_mode_(other.actual_decode_mode_),
      capture_(std::move(other.capture_)), width_(other.width_),
      height_(other.height_), fps_(other.fps_),
      current_frame_(other.current_frame_),
      connected_(other.connected_.load()) {
  other.width_ = 0;
  other.height_ = 0;
  other.fps_ = 0.0;
  other.current_frame_ = 0;
}

RtspSource &RtspSource::operator=(RtspSource &&other) noexcept {
  if (this != &other) {
    SourceNode::operator=(std::move(other));
    actual_decode_mode_ = other.actual_decode_mode_;
    capture_ = std::move(other.capture_);
    width_ = other.width_;
    height_ = other.height_;
    fps_ = other.fps_;
    current_frame_ = other.current_frame_;
    connected_ = other.connected_.load();

    other.width_ = 0;
    other.height_ = 0;
    other.fps_ = 0.0;
    other.current_frame_ = 0;
  }
  return *this;
}

void RtspSource::on_open() {
  if (config_.uri.empty()) {
    throw ConfigError("RTSP URI is empty");
  }

  if (config_.decode_mode == DecodeMode::GPU) {
    VP_LOG_WARN(
        "RtspSource '{}' GPU decode mode requested but falling back to CPU "
        "(RTSP GPU decode not fully supported yet)",
        name_);
  }

  capture_ = std::make_unique<cv::VideoCapture>();
  capture_->set(cv::CAP_PROP_BUFFERSIZE, 1);

  if (!capture_->open(config_.uri, cv::CAP_FFMPEG)) {
    throw StreamError("Failed to open RTSP stream: " + config_.uri);
  }

  width_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
  height_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
  fps_ = capture_->get(cv::CAP_PROP_FPS);
  if (fps_ <= 0) {
    fps_ = 25.0;
  }

  actual_decode_mode_ = DecodeMode::CPU;
  connected_ = true;
  current_frame_ = 0;

  VP_LOG_INFO("RtspSource '{}' opened: {}x{}, fps={}", name_, width_, height_,
              fps_);
}

bool RtspSource::read_next(Frame &frame) {
  if (!capture_ || !capture_->isOpened()) {
    return false;
  }

  cv::Mat cpu_frame;
  if (!capture_->read(cpu_frame)) {
    connected_ = false;
    return false;
  }

  frame.pts_us =
      static_cast<int64_t>(capture_->get(cv::CAP_PROP_POS_MSEC) * 1000);
  ++current_frame_;

  return true;
}

void RtspSource::on_close() {
  connected_ = false;
  capture_.reset();
}

void RtspSource::on_read_error(const std::exception &e) {
  VP_LOG_ERROR("RtspSource '{}' read error: {}", name_, e.what());
  connected_ = false;
}

} // namespace visionpipe
