#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/frame.h"
#include "core/infer_node.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

/// @brief RTMPose 节点配置
struct RtmPoseConfig {
    int input_width = 192;            ///< 模型输入宽度（RTMPose-s/m/l 256x192）
    int input_height = 256;           ///< 模型输入高度
    std::vector<int> target_classes = {0};  ///< 参与姿态估计的 detection 类别（默认 person）
    float score_threshold = 0.3f;     ///< 关键点置信度阈值（低于该值的关键点 score 保留但建议渲染层过滤）
    float bbox_padding = 1.25f;       ///< 裁剪框相对检测框的扩展比例（对齐 mmpose GetBBoxCenterScale）
    float simcc_split_ratio = 2.0f;   ///< SimCC 划分比例
    int max_batch_size = 16;          ///< 单帧最多处理的目标数（受 engine 动态 batch 上限约束）
    size_t workers = 1;               ///< 并行 worker 数量
};

/// @brief RTMPose top-down 关键点检测节点
///
/// 绑定 mmdeploy 导出的 RTMPose SimCC 格式：
/// - 输入: [batch, 3, input_h, input_w]，RGB，mean/std 归一化（ImageNet 常数）
/// - 双输出: simcc_x [batch, K, input_w*2] + simcc_y [batch, K, input_h*2]
///
/// 流程（top-down，依赖上游 DetectorNode 提供人体框）：
/// 1. 取 frame.detections 中 target_classes 的框，按 bbox_padding 扩展并对齐
///    模型输入纵横比
/// 2. warpAffine 裁剪 → batch 推理（动态 batch engine）
/// 3. SimCC 解码 → 映射回原图 → frame.poses（经 detection_index 关联）
///
/// 后续支持其他关键点模型（heatmap 系等）时，再从本类与新实现中提取
/// 公共基类，而非现在预设抽象接口。
class RtmPoseNode : public InferNode {
public:
    explicit RtmPoseNode(std::shared_ptr<IModelEngine> engine,
                         const RtmPoseConfig& config = RtmPoseConfig(),
                         const std::string& name = "rtmpose");

    explicit RtmPoseNode(std::shared_ptr<IModelEngine> engine,
                         const std::string& name);

    ~RtmPoseNode() override = default;

    RtmPoseNode(const RtmPoseNode&) = delete;
    RtmPoseNode& operator=(const RtmPoseNode&) = delete;
    RtmPoseNode(RtmPoseNode&&) = delete;
    RtmPoseNode& operator=(RtmPoseNode&&) = delete;

    /// @brief 设置参数（支持热更新: score_threshold / bbox_padding / max_batch_size）
    bool set_param(const std::string& name, const ParamValue& value) override;

    const RtmPoseConfig& config() const { return config_; }

protected:
    void process_batch(std::vector<Frame>& frames) override;

private:
    /// @brief 单个裁剪区域（原图像素空间，未裁剪到图像边界）
    struct CropRect {
        float x1, y1, w, h;
    };

    void infer_frame(Frame& frame);

    bool matches_target_class(int class_id) const;

    void get_image_dims(const Frame& frame, int& width, int& height) const;

    /// @brief 由检测框计算纵横比对齐 + padding 扩展后的裁剪区域
    CropRect compute_crop_rect(const Detection& det, int orig_w, int orig_h) const;

    /// @brief warpAffine 裁剪单个目标并归一化，输出 CHW float 数据
    void crop_and_preprocess(const Frame& frame, const CropRect& rect,
                             int orig_w, int orig_h,
                             std::vector<float>& crop_data) const;

    RtmPoseConfig config_;
};

using RtmPoseNodePtr = std::shared_ptr<RtmPoseNode>;

}  // namespace visionpipe
