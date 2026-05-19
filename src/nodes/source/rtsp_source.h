#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "core/source_node.h"
#include "nodes/source/source_config.h"

namespace cv {
class VideoCapture;
}

namespace visionpipe {

class RtspSource : public SourceNode {
public:
  explicit RtspSource(const SourceConfig &config);

  explicit RtspSource(const std::string &uri,
                      DecodeMode mode = DecodeMode::AUTO);

  ~RtspSource() override = default;

  RtspSource(const RtspSource &) = delete;
  RtspSource &operator=(const RtspSource &) = delete;

  RtspSource(RtspSource &&other) noexcept;
  RtspSource &operator=(RtspSource &&other) noexcept;

  int width() const { return width_; }
  int height() const { return height_; }
  double fps() const { return fps_; }
  int64_t current_frame() const { return current_frame_; }
  DecodeMode actual_decode_mode() const { return actual_decode_mode_; }
  bool is_connected() const { return connected_; }

protected:
  void on_open() override;
  bool read_next(Frame &frame) override;
  void on_close() override;
  void on_read_error(const std::exception &e) override;

private:
  DecodeMode actual_decode_mode_;

  std::unique_ptr<cv::VideoCapture> capture_;

  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;
  int64_t current_frame_ = 0;
  std::atomic<bool> connected_{false};
};

using RtspSourcePtr = std::shared_ptr<RtspSource>;

} // namespace visionpipe
