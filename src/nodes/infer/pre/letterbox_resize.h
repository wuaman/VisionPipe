#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <cuda_runtime_api.h>

#include "core/tensor.h"

namespace visionpipe {

/// @brief Letterbox resize 参数
struct LetterboxParams {
    int target_width = 640;    ///< 目标宽度
    int target_height = 640;   ///< 目标高度
    float scale = 1.0f;        ///< 缩放比例
    int pad_x = 0;             ///< 水平填充
    int pad_y = 0;             ///< 垂直填充
    cv::Scalar pad_color = cv::Scalar(114, 114, 114);  ///< 填充颜色
};

/// @brief Letterbox resize 预处理
///
/// 将任意尺寸图像缩放并填充到目标尺寸，保持纵横比。
/// 支持 CPU 和 GPU 两种实现。
class LetterboxResize {
public:
    /// @brief 计算 letterbox 参数
    /// @param src_width 源图像宽度
    /// @param src_height 源图像高度
    /// @param target_width 目标宽度
    /// @param target_height 目标高度
    /// @return Letterbox 参数
    static LetterboxParams compute_params(int src_width, int src_height,
                                          int target_width, int target_height);

    /// @brief CPU 实现：执行 letterbox resize
    /// @param src 源图像（CPU Mat）
    /// @param dst 目标图像（CPU Mat）
    /// @param params letterbox 参数
    static void compute_cpu(const cv::Mat& src, cv::Mat& dst,
                           const LetterboxParams& params);

    /// @brief GPU 实现：执行 letterbox resize
    /// @param src 源图像（GPU GpuMat）
    /// @param dst 目标图像（GPU GpuMat）
    /// @param params letterbox 参数
    /// @param stream CUDA stream（可选）
    static void compute_gpu(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                           const LetterboxParams& params,
                           cudaStream_t stream = 0);

    /// @brief GPU 实现：直接输出到 Tensor
    /// @param src 源图像（GPU GpuMat）
    /// @param output 输出 Tensor（CHW float32 格式）
    /// @param params letterbox 参数
    /// @param stream CUDA stream（可选）
    static void compute_gpu_to_tensor(const cv::cuda::GpuMat& src,
                                      Tensor& output,
                                      const LetterboxParams& params,
                                      cudaStream_t stream = 0);

    /// @brief 将检测结果坐标从 letterbox 空间映射回原图空间
    /// @param bbox 边界框 [x1, y1, x2, y2]，会被原地修改
    /// @param params letterbox 参数
    static void map_bbox_back(float bbox[4], const LetterboxParams& params);
};

}  // namespace visionpipe
