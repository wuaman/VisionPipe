#pragma once

#include <memory>
#include <string>

#include "core/infer_node.h"
#include "core/frame.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/post/seg_mask_decoder.h"

namespace visionpipe {

/// @brief YOLO-seg 分割节点配置
struct YoloSegConfig {
    int input_width = 640;           ///< 模型输入宽度
    int input_height = 640;          ///< 模型输入高度
    float score_threshold = 0.25f;   ///< 检测置信度阈值
    float nms_threshold = 0.45f;     ///< NMS IoU 阈值
    float mask_threshold = 0.5f;     ///< 掩码二值化阈值
    int max_detections = 100;        ///< 最大检测数量
    size_t workers = 1;              ///< 并行 worker 数量
};

/// @brief YOLO-seg 系列实例分割节点
///
/// 绑定 YOLOv8/YOLO11-seg 的 TRT 导出格式：
/// - 双输出：检测头 [1, 4+nc+32, anchors] + 原型掩码 [1, 32, mask_h, mask_w]
/// - cxcywh 框格式、32 个 mask coefficients
///
/// 实现实例分割的完整流程：
/// 1. Letterbox resize 预处理
/// 2. TensorRT 推理（双输出：检测 + 原型掩码）
/// 3. NMS 后处理
/// 4. Mask 解码和裁剪（letterbox 逆映射对齐原图）
///
/// 后续支持其他分割模型（如 Mask R-CNN）时，再从本类与新实现中
/// 提取公共基类，而非现在预设抽象接口。
class YoloSegNode : public InferNode {
public:
    /// @brief 构造函数
    explicit YoloSegNode(std::shared_ptr<IModelEngine> engine,
                         const YoloSegConfig& config = YoloSegConfig(),
                         const std::string& name = "yolo_seg");

    /// @brief 简化构造函数
    explicit YoloSegNode(std::shared_ptr<IModelEngine> engine,
                         const std::string& name);

    ~YoloSegNode() override = default;

    // 禁止拷贝和移动（包含 mutex 成员）
    YoloSegNode(const YoloSegNode&) = delete;
    YoloSegNode& operator=(const YoloSegNode&) = delete;
    YoloSegNode(YoloSegNode&&) = delete;
    YoloSegNode& operator=(YoloSegNode&&) = delete;

    /// @brief 设置参数（支持热更新）
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const YoloSegConfig& config() const { return config_; }

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

    YoloSegConfig config_;
    std::vector<std::vector<uint8_t>> last_masks_;
    mutable std::mutex masks_mutex_;
};

/// @brief YoloSegNode 智能指针类型
using YoloSegNodePtr = std::shared_ptr<YoloSegNode>;

}  // namespace visionpipe
