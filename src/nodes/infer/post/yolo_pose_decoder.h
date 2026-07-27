#pragma once

#include <vector>

#include "core/frame.h"
#include "core/tensor.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {

/// @brief YOLO-pose 解码参数
struct YoloPoseParams {
    float score_threshold = 0.25f;  ///< 人体检测置信度阈值
    float nms_threshold = 0.45f;    ///< NMS IoU 阈值
    int max_detections = 100;       ///< 最大检测数量
    int num_keypoints = 17;         ///< 关键点数（COCO-17）
    int input_width = 640;          ///< 模型输入宽度
    int input_height = 640;         ///< 模型输入高度
};

/// @brief YOLOv8/11-pose 输出解码器
///
/// 输出格式：[1, 4 + 1 + K*3, num_anchors]
/// - 4: bbox cxcywh（letterbox 空间像素）
/// - 1: 人体置信度（已 sigmoid）
/// - K*3: 每个关键点 (x, y, visibility)，x/y 为 letterbox 空间像素，
///   visibility 已 sigmoid
class YoloPoseDecoder {
public:
    /// @brief 解码 YOLO-pose 输出
    /// @param output 检测输出 tensor [1, 5+K*3, num_anchors]
    /// @param detections 输出检测（bbox 归一化到原图）
    /// @param poses 输出关键点（归一化到原图，detection_index 对齐 detections 下标）
    static void decode(const Tensor& output,
                       std::vector<Detection>& detections,
                       std::vector<PoseResult>& poses,
                       const YoloPoseParams& params,
                       const LetterboxParams& letterbox_params,
                       int orig_width, int orig_height);
};

}  // namespace visionpipe
