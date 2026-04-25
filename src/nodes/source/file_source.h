#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "core/frame.h"
#include "core/node_base.h"
#include "source_config.h"

// Include OpenCV headers for complete type definitions
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <opencv2/cudacodec.hpp>
#endif

namespace visionpipe {

/// @brief 文件视频源节点
///
/// 从本地视频文件读取帧，支持 GPU 硬解码（cv::cudacodec）和 CPU
/// 软解码（cv::VideoCapture）。 AUTO 模式：优先使用 GPU 硬解码，NVCUVID
/// 不可用时自动退化为 CPU。 GPU 模式：强制 GPU 硬解码，NVCUVID 不可用时抛出
/// CudaError。 CPU 模式：强制 CPU 软解码。
class FileSource : public NodeBase {
public:
  /// @brief 构造函数
  /// @param config 视频源配置
  explicit FileSource(const SourceConfig &config);

  /// @brief 简化构造函数
  /// @param uri 视频文件路径
  /// @param mode 解码模式，默认 AUTO
  explicit FileSource(const std::string &uri,
                      DecodeMode mode = DecodeMode::AUTO);

  /// @brief 析构函数
  ~FileSource() override;

  // 禁止拷贝
  FileSource(const FileSource &) = delete;
  FileSource &operator=(const FileSource &) = delete;

  // 允许移动
  FileSource(FileSource &&other) noexcept;
  FileSource &operator=(FileSource &&other) noexcept;

  /// @brief 处理帧（源节点不使用此接口）
  void process(Frame &frame) override;

  /// @brief 启动视频源
  void start() override;

  /// @brief 停止视频源
  void stop(bool drain = true) override;

  /// @brief 是否为源节点
  bool is_source() const override { return true; }

  /// @brief 获取视频宽度
  int width() const { return width_; }

  /// @brief 获取视频高度
  int height() const { return height_; }

  /// @brief 获取视频帧率
  double fps() const { return fps_; }

  /// @brief 获取总帧数（-1 表示未知，如 RTSP 流）
  int64_t frame_count() const { return frame_count_; }

  /// @brief 获取当前帧号
  int64_t current_frame() const { return current_frame_; }

  /// @brief 获取实际使用的解码模式
  DecodeMode actual_decode_mode() const { return actual_decode_mode_; }

  /// @brief 获取源配置
  const SourceConfig &config() const { return config_; }

protected:
  /// @brief 初始化视频源
  void on_init() override;

  /// @brief 停止前回调
  void on_stop() override;

  /// @brief 源节点工作线程主循环
  void source_worker_loop();

  /// @brief 尝试初始化 GPU 解码器
  /// @return 是否成功初始化
  bool try_init_gpu_decoder();

  /// @brief 初始化 CPU 解码器
  void init_cpu_decoder();

  /// @brief 读取下一帧（GPU 模式）
  /// @param frame 输出帧
  /// @return 是否成功读取
  bool read_frame_gpu(Frame &frame);

  /// @brief 读取下一帧（CPU 模式）
  /// @param frame 输出帧
  /// @return 是否成功读取
  bool read_frame_cpu(Frame &frame);

  /// @brief 检测 NVCUVID 是否可用
  static bool is_nvdec_available();

private:
  SourceConfig config_;
  DecodeMode actual_decode_mode_; ///< 实际使用的解码模式

  // OpenCV 视频捕获（CPU 模式）
  std::unique_ptr<cv::VideoCapture> cpu_capture_;

#ifdef VISIONPIPE_USE_CUDA
  // OpenCV CUDA 视频阅读器（GPU 模式）- 使用 cv::Ptr 因为工厂函数返回 cv::Ptr
  cv::Ptr<cv::cudacodec::VideoReader> gpu_reader_;
#endif

  // 视频属性
  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;
  int64_t frame_count_ = -1;

  // 当前帧计数器
  std::atomic<int64_t> current_frame_{0};

  // 源节点专用工作线程
  std::thread source_thread_;
};

/// @brief FileSource 智能指针类型
using FileSourcePtr = std::shared_ptr<FileSource>;

} // namespace visionpipe
