#include "classifier_node.h"

#include <algorithm>
#include <cmath>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime_api.h>

#include "core/error.h"
#include "core/logger.h"
#include "hal/nvidia/cuda_allocator.h"

namespace visionpipe {

namespace {

// ImageNet 标准归一化参数
constexpr float kMeanR = 0.485f;
constexpr float kMeanG = 0.456f;
constexpr float kMeanB = 0.406f;
constexpr float kStdR = 0.229f;
constexpr float kStdG = 0.224f;
constexpr float kStdB = 0.225f;

// 全局 CUDA allocator
std::shared_ptr<CudaAllocator> g_cuda_allocator;

CudaAllocator* get_cuda_allocator() {
    if (!g_cuda_allocator) {
        g_cuda_allocator = std::make_shared<CudaAllocator>();
    }
    return g_cuda_allocator.get();
}

}  // namespace

ClassifierNode::ClassifierNode(std::shared_ptr<IModelEngine> engine,
                               const ClassifierConfig& config,
                               const std::string& name)
    : InferNode(std::move(engine), config.workers, name)
    , config_(config) {}

ClassifierNode::ClassifierNode(std::shared_ptr<IModelEngine> engine,
                               const std::string& name)
    : ClassifierNode(std::move(engine), ClassifierConfig(), name) {}

void ClassifierNode::infer_frame(IExecContext& ctx, Frame& frame) {
    if (frame.detections.empty()) {
        return;  // passthrough — no inference needed
    }

    Tensor batch_tensor;
    std::vector<int> valid_crop_indices;
    preprocess(frame, batch_tensor, valid_crop_indices);

    if (valid_crop_indices.empty()) {
        return;  // passthrough — no valid crops
    }

    Tensor output;
    ctx.infer(batch_tensor, output);

    postprocess(frame, output, valid_crop_indices);
}

void ClassifierNode::preprocess(Frame& frame, Tensor& batch_tensor,
                                std::vector<int>& valid_crop_indices) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    valid_crop_indices.clear();

    int orig_width = 0;
    int orig_height = 0;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        if (frame.image.shape.size() == 3) {
            if (frame.image.shape[2] == 3) {
                orig_height = static_cast<int>(frame.image.shape[0]);
                orig_width = static_cast<int>(frame.image.shape[1]);
            } else if (frame.image.shape[0] == 3) {
                orig_height = static_cast<int>(frame.image.shape[1]);
                orig_width = static_cast<int>(frame.image.shape[2]);
            }
        }
    } else {
        if (frame.image.shape.size() == 3) {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    }

    if (orig_width <= 0 || orig_height <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    int batch_size = std::min(static_cast<int>(frame.detections.size()),
                              config_.max_batch_size);

    batch_tensor = Tensor({batch_size, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    std::vector<float> all_crops_data;
    all_crops_data.reserve(batch_size * 3 * config_.input_height * config_.input_width);

    for (int i = 0; i < batch_size; ++i) {
        const auto& det = frame.detections[i];

        std::vector<float> crop_data;
        if (crop_and_preprocess(frame, det, crop_data)) {
            valid_crop_indices.push_back(i);
            all_crops_data.insert(all_crops_data.end(),
                                  crop_data.begin(), crop_data.end());
        }
    }

    if (valid_crop_indices.empty()) {
        batch_tensor = Tensor();
        return;
    }

    int actual_batch = static_cast<int>(valid_crop_indices.size());
    if (actual_batch != batch_size) {
        batch_tensor = Tensor({actual_batch, 3, config_.input_height, config_.input_width},
                              DataType::FLOAT32, get_cuda_allocator());
    }

    cudaMemcpy(batch_tensor.data, all_crops_data.data(),
               all_crops_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

bool ClassifierNode::crop_and_preprocess(Frame& frame, const Detection& det,
                                         std::vector<float>& crop_data) {
    int orig_width = 0;
    int orig_height = 0;
    int cv_type = CV_8UC3;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        if (frame.image.shape.size() == 3 && frame.image.shape[2] == 3) {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    } else {
        if (frame.image.shape.size() == 3) {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    }

    int x1 = static_cast<int>(det.bbox[0] * orig_width);
    int y1 = static_cast<int>(det.bbox[1] * orig_height);
    int x2 = static_cast<int>(det.bbox[2] * orig_width);
    int y2 = static_cast<int>(det.bbox[3] * orig_height);

    x1 = std::max(0, std::min(x1, orig_width - 1));
    y1 = std::max(0, std::min(y1, orig_height - 1));
    x2 = std::max(x1 + 1, std::min(x2, orig_width));
    y2 = std::max(y1 + 1, std::min(y2, orig_height));

    int crop_width = x2 - x1;
    int crop_height = y2 - y1;

    if (crop_width <= 0 || crop_height <= 0) {
        return false;
    }

    crop_data.resize(3 * config_.input_height * config_.input_width);

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        cv::cuda::GpuMat gpu_image(orig_height, orig_width, cv_type, frame.image.data);

        cv::Rect roi(x1, y1, crop_width, crop_height);
        cv::cuda::GpuMat gpu_crop = gpu_image(roi);

        cv::cuda::GpuMat gpu_resized;
        cv::cuda::resize(gpu_crop, gpu_resized,
                         cv::Size(config_.input_width, config_.input_height),
                         0, 0, cv::INTER_LINEAR);

        cv::cuda::GpuMat gpu_rgb;
        cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);

        cv::cuda::GpuMat gpu_float;
        gpu_rgb.convertTo(gpu_float, CV_32F, 1.0 / 255.0);

        cv::Mat host_float;
        gpu_float.download(host_float);

        const int plane_size = config_.input_height * config_.input_width;
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
                float r = pixel[0];
                float g = pixel[1];
                float b = pixel[2];

                if (config_.normalize_mean_std) {
                    r = (r - kMeanR) / kStdR;
                    g = (g - kMeanG) / kStdG;
                    b = (b - kMeanB) / kStdB;
                }

                crop_data[0 * plane_size + h * config_.input_width + w] = r;
                crop_data[1 * plane_size + h * config_.input_width + w] = g;
                crop_data[2 * plane_size + h * config_.input_width + w] = b;
            }
        }
    } else {
        cv::Mat cpu_image(orig_height, orig_width, cv_type, frame.image.data);

        cv::Rect roi(x1, y1, crop_width, crop_height);
        cv::Mat cpu_crop = cpu_image(roi);

        cv::Mat resized;
        cv::resize(cpu_crop, resized,
                   cv::Size(config_.input_width, config_.input_height),
                   0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

        const int plane_size = config_.input_height * config_.input_width;
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                const cv::Vec3f& pixel = float_img.at<cv::Vec3f>(h, w);
                float r = pixel[0];
                float g = pixel[1];
                float b = pixel[2];

                if (config_.normalize_mean_std) {
                    r = (r - kMeanR) / kStdR;
                    g = (g - kMeanG) / kStdG;
                    b = (b - kMeanB) / kStdB;
                }

                crop_data[0 * plane_size + h * config_.input_width + w] = r;
                crop_data[1 * plane_size + h * config_.input_width + w] = g;
                crop_data[2 * plane_size + h * config_.input_width + w] = b;
            }
        }
    }

    return true;
}

void ClassifierNode::postprocess(Frame& frame, const Tensor& output,
                                 const std::vector<int>& valid_crop_indices) {
    if (output.data == nullptr || valid_crop_indices.empty()) {
        return;
    }

    int batch_size = static_cast<int>(valid_crop_indices.size());
    int num_classes = 1;
    if (output.shape.size() >= 2) {
        num_classes = static_cast<int>(output.shape[1]);
    } else if (output.shape.size() == 1) {
        num_classes = static_cast<int>(output.shape[0]) / batch_size;
    }

    std::vector<float> host_output(output.nbytes / sizeof(float));
    cudaMemcpy(host_output.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        int det_idx = valid_crop_indices[i];
        if (det_idx >= static_cast<int>(frame.detections.size())) {
            continue;
        }

        std::vector<float> logits(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            logits[c] = host_output[i * num_classes + c];
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            logits[c] = std::exp(logits[c] - max_logit);
            sum_exp += logits[c];
        }
        for (int c = 0; c < num_classes; ++c) {
            logits[c] /= sum_exp;
        }

        int best_class = 0;
        float best_prob = logits[0];
        for (int c = 1; c < num_classes; ++c) {
            if (logits[c] > best_prob) {
                best_prob = logits[c];
                best_class = c;
            }
        }

        frame.detections[det_idx].class_id = best_class;
        frame.detections[det_idx].confidence = best_prob;
    }
}

}  // namespace visionpipe
