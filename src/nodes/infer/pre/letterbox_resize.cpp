#include "nodes/infer/pre/letterbox_resize.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime_api.h>

namespace visionpipe {

LetterboxParams LetterboxResize::compute_params(int src_width, int src_height,
                                                 int target_width, int target_height) {
    LetterboxParams params;
    params.target_width = target_width;
    params.target_height = target_height;

    // 计算缩放比例，保持纵横比
    float scale_x = static_cast<float>(target_width) / src_width;
    float scale_y = static_cast<float>(target_height) / src_height;
    params.scale = std::min(scale_x, scale_y);

    // 计算缩放后的尺寸
    int scaled_width = static_cast<int>(std::round(src_width * params.scale));
    int scaled_height = static_cast<int>(std::round(src_height * params.scale));

    // 计算填充
    params.pad_x = (target_width - scaled_width) / 2;
    params.pad_y = (target_height - scaled_height) / 2;

    return params;
}

void LetterboxResize::compute_cpu(const cv::Mat& src, cv::Mat& dst,
                                   const LetterboxParams& params) {
    // 缩放图像
    cv::Mat scaled;
    cv::resize(src, scaled, cv::Size(), params.scale, params.scale, cv::INTER_LINEAR);

    // 创建目标图像并填充
    dst.create(params.target_height, params.target_width, src.type());
    dst.setTo(params.pad_color);

    // 将缩放后的图像复制到目标位置
    cv::Rect roi(params.pad_x, params.pad_y, scaled.cols, scaled.rows);
    scaled.copyTo(dst(roi));
}

void LetterboxResize::compute_gpu(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                                   const LetterboxParams& params,
                                   cudaStream_t cuda_stream) {
    // 使用 OpenCV CUDA Stream
    cv::cuda::Stream stream;

    // 计算缩放后的目标大小
    int scaled_width = static_cast<int>(std::round(src.cols * params.scale));
    int scaled_height = static_cast<int>(std::round(src.rows * params.scale));

    // 缩放图像到明确的尺寸
    cv::cuda::GpuMat scaled(scaled_height, scaled_width, src.type());
    cv::cuda::resize(src, scaled, cv::Size(scaled_width, scaled_height), 0, 0,
                     cv::INTER_LINEAR, stream);

    // 创建目标图像并填充
    dst.create(params.target_height, params.target_width, src.type());
    dst.setTo(params.pad_color, stream);

    // 将缩放后的图像复制到目标位置
    cv::Rect roi(params.pad_x, params.pad_y, scaled.cols, scaled.rows);
    scaled.copyTo(dst(roi), stream);

    // 等待完成（如果提供了 CUDA stream，同步）
    stream.waitForCompletion();
    if (cuda_stream != 0) {
        cudaStreamSynchronize(cuda_stream);
    }
}

void LetterboxResize::compute_gpu_to_tensor(const cv::cuda::GpuMat& src,
                                             Tensor& output,
                                             const LetterboxParams& params,
                                             cudaStream_t cuda_stream) {
    // 执行 letterbox resize
    cv::cuda::GpuMat resized;
    compute_gpu(src, resized, params, cuda_stream);

    // 转换为 CHW float32 格式并归一化
    // YOLOv8 输入格式：[1, 3, H, W]，值范围 [0, 1]
    // OpenCV 格式：[H, W, 3] BGR

    const int height = resized.rows;
    const int width = resized.cols;
    const int channels = resized.channels();

    // 确保 output tensor 正确分配
    if (output.shape != std::vector<int64_t>{1, channels, height, width} ||
        output.dtype != DataType::FLOAT32 ||
        !output.valid()) {
        // 需要重新分配 tensor
        output = Tensor({1, channels, height, width}, DataType::FLOAT32, output.allocator);
    }

    cv::cuda::Stream stream;

    // BGR -> RGB
    cv::cuda::GpuMat rgb;
    cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB, 0, stream);

    // 转换为 float 并归一化
    cv::cuda::GpuMat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0, stream);

    stream.waitForCompletion();

    // HWC -> CHW：下载到 CPU 进行转换
    cv::Mat host_float;
    float_img.download(host_float);

    // 复制到输出 tensor（CHW 格式）
    float* output_ptr = static_cast<float*>(output.data);
    const size_t plane_size = height * width;

    // 在 CPU 上转换 HWC -> CHW，然后上传到 GPU
    std::vector<float> host_chw(3 * plane_size);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
            host_chw[0 * plane_size + h * width + w] = pixel[0];  // R
            host_chw[1 * plane_size + h * width + w] = pixel[1];  // G
            host_chw[2 * plane_size + h * width + w] = pixel[2];  // B
        }
    }

    // 上传到 GPU
    cudaMemcpy(output.data, host_chw.data(), output.nbytes, cudaMemcpyHostToDevice);
}

void LetterboxResize::map_bbox_back(float bbox[4], const LetterboxParams& params) {
    // bbox 格式：[x1, y1, x2, y2]
    // 从 letterbox 空间映射回原图空间
    bbox[0] = (bbox[0] - params.pad_x) / params.scale;  // x1
    bbox[1] = (bbox[1] - params.pad_y) / params.scale;  // y1
    bbox[2] = (bbox[2] - params.pad_x) / params.scale;  // x2
    bbox[3] = (bbox[3] - params.pad_y) / params.scale;  // y2
}

}  // namespace visionpipe