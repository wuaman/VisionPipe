#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "core/frame.h"
#include "core/infer_node.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {

/// @brief YOLO-pose 节点配置
struct YoloPoseConfig {
    int input_width = 640;          ///< 模型输入宽度
    int input_height = 640;         ///< 模型输入高度
    float score_threshold = 0.25f;  ///< 人体检测置信度阈值
    float nms_threshold = 0.45f;    ///< NMS IoU 阈值
    int max_detections = 100;       ///< 最大检测数量
    int num_keypoints = 17;         ///< 关键点数（COCO-17）
    size_t workers = 1;             ///< 并行 worker 数量
};

/// @brief YOLO-pose 系列单阶段关键点检测节点
///
/// 绑定 YOLOv8/YOLO11-pose 的 TRT 导出格式：
/// - 单输出：[1, 4+1+K*3, anchors]，bbox cxcywh + 人体置信度 + K 个 (x,y,vis)
///
/// 单阶段：一次推理同时产出 frame.detections（person）与 frame.poses，
/// 不依赖上游检测节点，人数多时延迟恒定（对比 top-down 的 RtmPoseNode）。
class YoloPoseNode : public InferNode {
public:
    explicit YoloPoseNode(std::shared_ptr<IModelEngine> engine,
                          const YoloPoseConfig& config = YoloPoseConfig(),
                          const std::string& name = "yolo_pose");

    explicit YoloPoseNode(std::shared_ptr<IModelEngine> engine,
                          const std::string& name);

    ~YoloPoseNode() override = default;

    YoloPoseNode(const YoloPoseNode&) = delete;
    YoloPoseNode& operator=(const YoloPoseNode&) = delete;
    YoloPoseNode(YoloPoseNode&&) = delete;
    YoloPoseNode& operator=(YoloPoseNode&&) = delete;

    /// @brief 设置参数（支持热更新: score_threshold / nms_threshold / max_detections）
    bool set_param(const std::string& name, const ParamValue& value) override;

    const YoloPoseConfig& config() const { return config_; }

protected:
    void process_batch(std::vector<Frame>& frames) override;

private:
    LetterboxParams preprocess(Frame& frame, Tensor& input_tensor);

    YoloPoseConfig config_;
};

using YoloPoseNodePtr = std::shared_ptr<YoloPoseNode>;

}  // namespace visionpipe
