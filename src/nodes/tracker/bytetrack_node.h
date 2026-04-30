#pragma once

#include <memory>
#include <string>

#include "core/node_base.h"
#include "core/frame.h"
#include "nodes/tracker/bytetrack_impl.h"

namespace visionpipe {

/// @brief ByteTrack 追踪节点配置
struct ByteTrackConfig {
    float track_thresh = 0.5f;     ///< 追踪置信度阈值
    int track_buffer = 30;         ///< 追踪缓冲帧数
    float match_thresh = 0.3f;     ///< 匹配 IoU 阈值
    int frame_rate = 30;           ///< 帧率
};

/// @brief ByteTrack 多目标追踪节点
///
/// 纯 CPU 实现，接收 DetectorNode 输出的检测结果，
/// 执行多目标追踪，将追踪 ID 写入 Detection::track_id。
///
/// 工作流程：
/// 1. 读取 frame.detections
/// 2. 执行 ByteTrack 算法
/// 3. 更新 frame.detections[i].track_id
/// 4. 将活跃轨迹写入 frame.tracks
class ByteTrackNode : public NodeBase {
public:
    /// @brief 构造函数
    /// @param config 追踪配置
    /// @param name 节点名称
    explicit ByteTrackNode(const ByteTrackConfig& config = ByteTrackConfig(),
                           const std::string& name = "bytetrack");

    /// @brief 析构函数
    ~ByteTrackNode() override;

    // 禁止拷贝
    ByteTrackNode(const ByteTrackNode&) = delete;
    ByteTrackNode& operator=(const ByteTrackNode&) = delete;

    // 禁止移动（包含 unique_ptr 成员）
    ByteTrackNode(ByteTrackNode&&) = delete;
    ByteTrackNode& operator=(ByteTrackNode&&) = delete;

    /// @brief 处理帧
    void process(Frame& frame) override;

    /// @brief 设置参数（支持热更新）
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const ByteTrackConfig& config() const { return config_; }

    /// @brief 重置追踪器
    void reset();

    /// @brief 获取当前活跃轨迹数
    size_t active_track_count() const;

private:
    ByteTrackConfig config_;
    std::unique_ptr<ByteTrackImpl> tracker_;
};

/// @brief ByteTrackNode 智能指针类型
using ByteTrackNodePtr = std::shared_ptr<ByteTrackNode>;

}  // namespace visionpipe
