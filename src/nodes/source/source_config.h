#pragma once

#include "core/bounded_queue.h"
#include <string>

namespace visionpipe {

/// @brief 解码模式
enum class DecodeMode {
  AUTO, ///< 自动检测：优先 GPU 硬解，不可用时退化为 CPU
  GPU,  ///< 强制 GPU 硬解（cv::cudacodec），不可用时抛异常
  CPU   ///< 强制 CPU 软解（cv::VideoCapture）
};

/// @brief 视频源配置
struct SourceConfig {
  std::string uri; ///< 文件路径、RTSP URL、设备号
  DecodeMode decode_mode = DecodeMode::AUTO; ///< 解码模式
  int gpu_device = 0;         ///< GPU 设备号（多卡场景）
  size_t queue_capacity = 16; ///< 输出队列容量
  OverflowPolicy overflow_policy = OverflowPolicy::DROP_OLDEST; ///< 溢出策略
  int64_t stream_id = 0; ///< 流标识符

  /// @brief 默认构造函数
  SourceConfig() = default;

  /// @brief 简化构造函数（仅 URI）
  explicit SourceConfig(const std::string &uri_) : uri(uri_) {}

  /// @brief 完整构造函数
  SourceConfig(const std::string &uri_, DecodeMode mode, int gpu_dev = 0,
               size_t cap = 16,
               OverflowPolicy policy = OverflowPolicy::DROP_OLDEST,
               int64_t sid = 0)
      : uri(uri_), decode_mode(mode), gpu_device(gpu_dev), queue_capacity(cap),
        overflow_policy(policy), stream_id(sid) {}
};

} // namespace visionpipe