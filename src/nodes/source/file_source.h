#pragma once

#include <memory>
#include <string>

#include "core/source_node.h"
#include "nodes/source/source_config.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <opencv2/cudacodec.hpp>
#endif

namespace visionpipe {

class FileSource : public SourceNode {
public:
  explicit FileSource(const SourceConfig &config);

  explicit FileSource(const std::string &uri,
                      DecodeMode mode = DecodeMode::AUTO);

  ~FileSource() override = default;

  FileSource(const FileSource &) = delete;
  FileSource &operator=(const FileSource &) = delete;

  FileSource(FileSource &&other) noexcept;
  FileSource &operator=(FileSource &&other) noexcept;

  int width() const { return width_; }
  int height() const { return height_; }
  double fps() const { return fps_; }
  int64_t frame_count() const { return frame_count_; }
  int64_t current_frame() const { return current_frame_; }
  DecodeMode actual_decode_mode() const { return actual_decode_mode_; }

protected:
  void on_open() override;
  bool read_next(Frame &frame) override;
  void on_close() override;

private:
  bool try_init_gpu_decoder();
  void init_cpu_decoder();
  bool read_frame_gpu(Frame &frame);
  bool read_frame_cpu(Frame &frame);
  static bool is_nvdec_available();

  DecodeMode actual_decode_mode_;

  std::unique_ptr<cv::VideoCapture> cpu_capture_;

#ifdef VISIONPIPE_USE_CUDA
  cv::Ptr<cv::cudacodec::VideoReader> gpu_reader_;
#endif

  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;
  int64_t frame_count_ = -1;
  int64_t current_frame_ = 0;
};

using FileSourcePtr = std::shared_ptr<FileSource>;

} // namespace visionpipe
