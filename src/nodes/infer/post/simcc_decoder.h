#pragma once

#include <vector>

#include "core/frame.h"

namespace visionpipe {

/// @brief RTMPose SimCC 输出解码器
///
/// SimCC 将关键点定位建模为 x/y 两个独立的一维分类问题：
/// - simcc_x: [batch, K, input_w * split_ratio]
/// - simcc_y: [batch, K, input_h * split_ratio]
///
/// 解码：对每个关键点取两个向量的 argmax / split_ratio 得到模型输入
/// 空间坐标，置信度取两轴最大响应的均值。
class SimccDecoder {
public:
    /// @brief 解码单个目标（batch 内一个样本）的 SimCC 输出
    /// @param simcc_x x 轴分类向量，[K, x_bins] 行优先
    /// @param simcc_y y 轴分类向量，[K, y_bins] 行优先
    /// @param num_keypoints 关键点数 K
    /// @param x_bins x 轴 bin 数（= input_w * split_ratio）
    /// @param y_bins y 轴 bin 数（= input_h * split_ratio）
    /// @param split_ratio SimCC 划分比例（RTMPose 默认 2.0）
    /// @param keypoints 输出关键点，坐标为模型输入空间像素
    static void decode(const float* simcc_x, const float* simcc_y,
                       int num_keypoints, int x_bins, int y_bins,
                       float split_ratio,
                       std::vector<Keypoint>& keypoints);
};

}  // namespace visionpipe
