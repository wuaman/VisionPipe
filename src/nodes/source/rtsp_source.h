#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "core/frame.h"
#include "core/node_base.h"
#include "source_config.h"

// Forward declarations for OpenCV types
namespace cv {
class VideoCapture;
}

namespace visionpipe {

/// @brief RTSP 视频源节点
///
/// 从 RTSP 流读取帧，支持 GPU 硬解码和 CPU 软解码。
/// AUTO 模式：优先使用 GPU 硬解码，NVCUVID 不可用时自动退化为 CPU。
/// GPU 模式：强制 GPU 硬解码，NVCUVID 不可用时抛出 CudaError。
/// CPU 模式：强制 CPU 软解码。
///
/// RTSP 流特点：
/// - 无固定帧数（frame_count = -1）
/// - 可能需要重连机制
/// - 延迟可能较高
class RtspSource : public NodeBase {
public:
  /// @brief 构造函数
  /// @param config 视频源配置
  explicit RtspSource(const SourceConfig &config);

  /// @brief 简化构造函数
  /// @param uri RTSP URL
  /// @param mode 解码模式，默认 AUTO
  explicit RtspSource(const std::string &uri,
                      DecodeMode mode = DecodeMode::AUTO);

  /// @brief 析构函数
  ~RtspSource() override;

  // 禁止拷贝
  RtspSource(const RtspSource &) = delete;
  RtspSource &operator=(const RtspSource &) = delete;

  // 允许移动
  RtspSource(RtspSource &&other) noexcept;
  RtspSource &operator=(RtspSource &&other) noexcept;

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

  /// @brief 获取当前帧号
  int64_t current_frame() const { return current_frame_; }

  /// @brief 获取实际使用的解码模式
  DecodeMode actual_decode_mode() const { return actual_decode_mode_; }

  /// @brief 获取源配置
  const SourceConfig &config() const { return config_; }

  /// @brief 获取连接状态
  bool is_connected() const { return connected_; }

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

  /// @brief 读取下一帧
  /// @param frame 输出帧
  /// @return 是否成功读取
  bool read_frame(Frame &frame);

  /// @brief 尝试重连
  /// @return 是否成功重连
  bool try_reconnect();

private:
  SourceConfig config_;
  DecodeMode actual_decode_mode_; ///< 实际使用的解码模式

  // OpenCV 视频捕获
  std::unique_ptr<cv::VideoCapture> capture_;

  // 视频属性
  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;

  // 当前帧计数器
  std::atomic<int64_t> current_frame_{0};

  // 连接状态
  std::atomic<bool> connected_{false};

  // 源节点专用工作线程
  std::thread source_thread_;

  // 重连参数
  static constexpr int max_reconnect_attempts = 5;
  static constexpr int reconnect_delay_ms = 1000;
};

/// @brief RtspSource 智能指针类型
using RtspSourcePtr = std::shared_ptr<RtspSource>;

} // namespace visionpipe