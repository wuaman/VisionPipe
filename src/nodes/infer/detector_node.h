#pragma once

#include <memory>
#include <string>

#include "core/infer_node.h"
#include "core/frame.h"
#include "nodes/infer/pre/letterbox_resize.h"
#include "nodes/infer/post/detection_decoder.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

/// @brief 检测节点配置
struct DetectorConfig {
    int input_width = 640;           ///< 模型输入宽度
    int input_height = 640;          ///< 模型输入高度
    float score_threshold = 0.25f;   ///< 置信度阈值
    float nms_threshold = 0.45f;     ///< NMS IoU 阈值
    int max_detections = 300;        ///< 最大检测数量
    size_t workers = 1;              ///< 并行 worker 数量
};

/// @brief YOLOv8 检测节点
///
/// 实现目标检测的完整流程：
/// 1. Letterbox resize 预处理
/// 2. TensorRT 推理
/// 3. NMS 后处理
/// 4. 坐标映射回原图空间
class DetectorNode : public InferNode {
public:
    /// @brief 构造函数
    /// @param engine TensorRT 模型引擎
    /// @param config 检测配置
    /// @param name 节点名称
    explicit DetectorNode(std::shared_ptr<IModelEngine> engine,
                          const DetectorConfig& config = DetectorConfig(),
                          const std::string& name = "detector");

    /// @brief 简化构造函数
    /// @param engine TensorRT 模型引擎
    /// @param name 节点名称
    explicit DetectorNode(std::shared_ptr<IModelEngine> engine,
                          const std::string& name);

    ~DetectorNode() override = default;

    // 禁止拷贝
    DetectorNode(const DetectorNode&) = delete;
    DetectorNode& operator=(const DetectorNode&) = delete;

    // 允许移动
    DetectorNode(DetectorNode&&) noexcept = default;
    DetectorNode& operator=(DetectorNode&&) noexcept = default;

    /// @brief 设置参数（支持热更新）
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const DetectorConfig& config() const { return config_; }

    /// @brief 设置 ROI（感兴趣区域）
    /// @param polygons ROI 多边形顶点列表，坐标归一化到 [0, 1]
    void set_roi(const std::vector<std::vector<float>>& polygons);

    /// @brief 清除 ROI
    void clear_roi();

protected:
    void infer_frame(IExecContext& ctx, Frame& frame) override;

private:
    /// @brief 预处理图像
    LetterboxParams preprocess(Frame& frame, Tensor& input_tensor);

    /// @brief 后处理推理结果
    void postprocess(Frame& frame, const Tensor& output,
                     const LetterboxParams& letterbox_params,
                     int orig_width, int orig_height);

    /// @brief 检查检测结果是否在 ROI 内
    bool is_in_roi(const Detection& det) const;

    DetectorConfig config_;
    std::vector<std::vector<cv::Point2f>> roi_polygons_;
    mutable std::mutex roi_mutex_;
};

/// @brief DetectorNode 智能指针类型
using DetectorNodePtr = std::shared_ptr<DetectorNode>;

}  // namespace visionpipe
