#pragma once

#include <vector>

#include "core/frame.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {

/// @brief NMS 参数
struct NmsParams {
    float score_threshold = 0.25f;  ///< 置信度阈值
    float iou_threshold = 0.45f;    ///< NMS IoU 阈值
    int max_detections = 300;       ///< 最大检测数量
};

/// @brief YOLOv8 检测解码器
///
/// 解析 YOLOv8 TensorRT 输出 tensor 并执行 NMS 后处理。
/// YOLOv8 输出格式：[1, 84, 8400]（对于 640x640 输入）
/// - 84 = 4 (bbox: cx, cy, w, h) + 80 (class scores)
/// - 8400 = 预测框数量（来自不同尺度的特征图）
class DetectionDecoder {
public:
    /// @brief 解码 YOLOv8 输出
    /// @param output TensorRT 输出 tensor
    /// @param detections 检测结果列表
    /// @param nms_params NMS 参数
    /// @param letterbox_params letterbox 参数（用于坐标映射）
    /// @param orig_width 原图宽度
    /// @param orig_height 原图高度
    static void decode(const Tensor& output,
                       std::vector<Detection>& detections,
                       const NmsParams& nms_params,
                       const LetterboxParams& letterbox_params,
                       int orig_width, int orig_height);

    /// @brief NMS 实现（CPU）
    /// @param detections 检测结果列表（会被原地修改）
    /// @param iou_threshold IoU 阈值
    static void nms(std::vector<Detection>& detections, float iou_threshold);

private:
    /// @brief 计算 IoU
    static float compute_iou(const float bbox1[4], const float bbox2[4]);

    /// @brief 将中心坐标转换为角点坐标
    static void cxcywh_to_xyxy(float bbox[4]);

    /// @brief 将检测框坐标从模型空间映射到原图空间
    static void scale_bbox(float bbox[4],
                          const LetterboxParams& params,
                          int orig_width, int orig_height);
};

}  // namespace visionpipe