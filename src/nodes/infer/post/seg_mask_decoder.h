#pragma once

#include <vector>

#include "core/frame.h"
#include "nodes/infer/pre/letterbox_resize.h"

namespace visionpipe {

/// @brief 分割掩码解码参数
struct SegMaskParams {
    float mask_threshold = 0.5f;    ///< 掩码二值化阈值
    float score_threshold = 0.25f;  ///< 检测置信度阈值
    float nms_threshold = 0.45f;    ///< NMS IoU 阈值
    int max_detections = 100;       ///< 最大检测数量
};

/// @brief YOLOv8-seg 分割掩码解码器
///
/// 解析 YOLOv8-seg TensorRT 输出并生成分割掩码。
///
/// YOLOv8-seg 输出格式：
/// - Output 0: 检测输出 [1, 84 + num_masks, num_anchors]
///   - 84 = 4 (bbox) + 80 (classes)
///   - num_masks = 32 (mask coefficients)
/// - Output 1: 原型掩码 [1, num_masks, mask_h, mask_w]
///
/// 分割流程：
/// 1. 解码检测框和 mask coefficients
/// 2. 执行 NMS
/// 3. 对每个检测框，计算 mask = sigmoid(proto @ coeffs)
/// 4. 将 mask 裁剪到检测框范围
class SegMaskDecoder {
public:
    /// @brief 解码 YOLOv8-seg 输出
    /// @param det_output 检测输出 tensor [1, 84+num_masks, num_anchors]
    /// @param proto_output 原型掩码 tensor [1, num_masks, mask_h, mask_w]
    /// @param detections 检测结果列表（会被填充）
    /// @param masks 输出掩码列表（每个检测对应一个二值掩码）
    /// @param params 解码参数
    /// @param letterbox_params letterbox 参数（用于坐标映射）
    /// @param orig_width 原图宽度
    /// @param orig_height 原图高度
    static void decode(const Tensor& det_output,
                       const Tensor& proto_output,
                       std::vector<Detection>& detections,
                       std::vector<std::vector<uint8_t>>& masks,
                       const SegMaskParams& params,
                       const LetterboxParams& letterbox_params,
                       int orig_width, int orig_height);

    /// @brief 计算 mask 与 bbox 的 IoU
    /// @param mask 二值掩码
    /// @param bbox 边界框 [x1, y1, x2, y2]，归一化坐标
    /// @param mask_width 掩码宽度
    /// @param mask_height 掩码高度
    /// @return IoU 值
    static float compute_mask_bbox_iou(const std::vector<uint8_t>& mask,
                                       const float bbox[4],
                                       int mask_width, int mask_height);

private:
    /// @brief 解码检测框和 mask coefficients
    /// @param det_output 检测输出 tensor
    /// @param params 解码参数
    /// @param bbox_list 输出边界框列表
    /// @param class_ids 输出类别 ID 列表
    /// @param confidences 输出置信度列表
    /// @param mask_coeffs 输出 mask coefficients 列表
    static void decode_detections(const Tensor& det_output,
                                  const SegMaskParams& params,
                                  std::vector<std::vector<float>>& bbox_list,
                                  std::vector<int>& class_ids,
                                  std::vector<float>& confidences,
                                  std::vector<std::vector<float>>& mask_coeffs);

    /// @brief 执行 NMS
    static void nms(std::vector<std::vector<float>>& bbox_list,
                    std::vector<int>& class_ids,
                    std::vector<float>& confidences,
                    std::vector<std::vector<float>>& mask_coeffs,
                    float iou_threshold,
                    int max_detections);

    /// @brief 计算 mask
    /// @param proto 原型掩码数据 [num_masks, mask_h, mask_w]
    /// @param coeffs mask coefficients [num_masks]
    /// @param mask 输出掩码
    /// @param mask_width 掩码宽度
    /// @param mask_height 掩码高度
    /// @param threshold 二值化阈值
    static void compute_mask(const float* proto,
                             const std::vector<float>& coeffs,
                             std::vector<uint8_t>& mask,
                             int mask_width, int mask_height,
                             float threshold);

    /// @brief 裁剪 mask 到 bbox 范围
    static void crop_mask_to_bbox(std::vector<uint8_t>& mask,
                                  const float bbox[4],
                                  int mask_width, int mask_height);

    /// @brief 缩放 bbox 从模型空间到原图空间
    static void scale_bbox(float bbox[4],
                          const LetterboxParams& params,
                          int orig_width, int orig_height);

    /// @brief 缩放 mask 到原图尺寸
    static void resize_mask(std::vector<uint8_t>& mask,
                            int src_width, int src_height,
                            int dst_width, int dst_height);
};

}  // namespace visionpipe
