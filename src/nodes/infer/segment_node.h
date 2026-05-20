#pragma once

#include <memory>
#include <string>

#include "core/infer_node.h"
#include "core/frame.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/post/seg_mask_decoder.h"

namespace visionpipe {

/// @brief 分割节点配置
struct SegmentConfig {
    int input_width = 640;           ///< 模型输入宽度
    int input_height = 640;          ///< 模型输入高度
    float score_threshold = 0.25f;   ///< 检测置信度阈值
    float nms_threshold = 0.45f;     ///< NMS IoU 阈值
    float mask_threshold = 0.5f;     ///< 掩码二值化阈值
    int max_detections = 100;        ///< 最大检测数量
    size_t workers = 1;              ///< 并行 worker 数量
};

/// @brief YOLOv8-seg 实例分割节点
///
/// 实现实例分割的完整流程：
/// 1. Letterbox resize 预处理
/// 2. TensorRT 推理（双输出：检测 + 原型掩码）
/// 3. NMS 后处理
/// 4. Mask 解码和裁剪
class SegmentNode : public InferNode {
public:
    /// @brief 构造函数
    explicit SegmentNode(std::shared_ptr<IModelEngine> engine,
                         const SegmentConfig& config = SegmentConfig(),
                         const std::string& name = "segment");

    /// @brief 简化构造函数
    explicit SegmentNode(std::shared_ptr<IModelEngine> engine,
                         const std::string& name);

    ~SegmentNode() override = default;

    // 禁止拷贝和移动（包含 mutex 成员）
    SegmentNode(const SegmentNode&) = delete;
    SegmentNode& operator=(const SegmentNode&) = delete;
    SegmentNode(SegmentNode&&) = delete;
    SegmentNode& operator=(SegmentNode&&) = delete;

    /// @brief 设置参数（支持热更新）
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const SegmentConfig& config() const { return config_; }

    /// @brief 获取最近一帧的分割掩码
    const std::vector<std::vector<uint8_t>>& last_masks() const { return last_masks_; }

protected:
    void process_batch(std::vector<Frame>& frames) override;

private:
    LetterboxParams preprocess(Frame& frame, Tensor& input_tensor);

    void postprocess(Frame& frame, const Tensor& det_output,
                     const Tensor& proto_output,
                     const LetterboxParams& letterbox_params,
                     int orig_width, int orig_height);

    SegmentConfig config_;
    std::vector<std::vector<uint8_t>> last_masks_;
    mutable std::mutex masks_mutex_;
};

/// @brief SegmentNode 智能指针类型
using SegmentNodePtr = std::shared_ptr<SegmentNode>;

}  // namespace visionpipe
