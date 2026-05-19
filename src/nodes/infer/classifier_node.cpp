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

constexpr float kMeanR = 0.485f;
constexpr float kMeanG = 0.456f;
constexpr float kMeanB = 0.406f;
constexpr float kStdR = 0.229f;
constexpr float kStdG = 0.224f;
constexpr float kStdB = 0.225f;

std::shared_ptr<CudaAllocator> g_cuda_allocator;

CudaAllocator* get_cuda_allocator() {
    if (!g_cuda_allocator) {
        g_cuda_allocator = std::make_shared<CudaAllocator>();
    }
    return g_cuda_allocator.get();
}

void extract_hwc_planes(const cv::Mat& float_img, std::vector<float>& out,
                         int h, int w, bool normalize,
                         float mr, float mg, float mb,
                         float sr, float sg, float sb) {
    const int plane_size = h * w;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& pixel = float_img.at<cv::Vec3f>(y, x);
            float r = pixel[0], g = pixel[1], b = pixel[2];
            if (normalize) {
                r = (r - mr) / sr;
                g = (g - mg) / sg;
                b = (b - mb) / sb;
            }
            out[0 * plane_size + y * w + x] = r;
            out[1 * plane_size + y * w + x] = g;
            out[2 * plane_size + y * w + x] = b;
        }
    }
}

int argmax_softmax(const float* logits, int num_classes, float& out_prob) {
    float max_logit = *std::max_element(logits, logits + num_classes);
    float sum_exp = 0.0f;
    std::vector<float> probs(num_classes);
    for (int c = 0; c < num_classes; ++c) {
        probs[c] = std::exp(logits[c] - max_logit);
        sum_exp += probs[c];
    }
    int best = 0;
    float best_p = probs[0] / sum_exp;
    for (int c = 1; c < num_classes; ++c) {
        float p = probs[c] / sum_exp;
        if (p > best_p) {
            best_p = p;
            best = c;
        }
    }
    out_prob = best_p;
    return best;
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

bool ClassifierNode::matches_target_class(int class_id) const {
    return std::find(config_.target_classes.begin(),
                     config_.target_classes.end(),
                     class_id) != config_.target_classes.end();
}

void ClassifierNode::get_image_dims(const Frame& frame, int& width, int& height) const {
    width = 0;
    height = 0;
    if (frame.image.shape.size() == 3) {
        if (frame.image.memory_type() == MemoryType::CUDA_DEVICE &&
            frame.image.shape[0] == 3) {
            height = static_cast<int>(frame.image.shape[1]);
            width = static_cast<int>(frame.image.shape[2]);
        } else {
            height = static_cast<int>(frame.image.shape[0]);
            width = static_cast<int>(frame.image.shape[1]);
        }
    }
}

void ClassifierNode::infer_frame(IExecContext& ctx, Frame& frame) {
    if (config_.target_classes.empty()) {
        infer_whole_image(ctx, frame);
    } else {
        infer_crops(ctx, frame);
    }
}

// Mode 2: whole-image classification
void ClassifierNode::infer_whole_image(IExecContext& ctx, Frame& frame) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    Tensor input_tensor;
    preprocess_whole_image(frame, input_tensor);

    Tensor output;
    ctx.infer(input_tensor, output);

    postprocess_whole_image(frame, output);
}

// Mode 1: crop-based secondary classification
void ClassifierNode::infer_crops(IExecContext& ctx, Frame& frame) {
    if (frame.detections.empty()) {
        return;
    }

    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    Tensor batch_tensor;
    std::vector<int> valid_det_indices;
    preprocess_crops(frame, batch_tensor, valid_det_indices);

    if (valid_det_indices.empty()) {
        return;
    }

    Tensor output;
    ctx.infer(batch_tensor, output);

    postprocess_crops(frame, output, valid_det_indices);
}

void ClassifierNode::preprocess_whole_image(Frame& frame, Tensor& input_tensor) {
    int orig_width = 0, orig_height = 0;
    get_image_dims(frame, orig_width, orig_height);
    if (orig_width <= 0 || orig_height <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    input_tensor = Tensor({1, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    std::vector<float> host_data(3 * config_.input_height * config_.input_width);

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        cv::cuda::GpuMat gpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);
        cv::cuda::GpuMat gpu_resized;
        cv::cuda::resize(gpu_image, gpu_resized,
                         cv::Size(config_.input_width, config_.input_height),
                         0, 0, cv::INTER_LINEAR);
        cv::cuda::GpuMat gpu_rgb;
        cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);
        cv::cuda::GpuMat gpu_float;
        gpu_rgb.convertTo(gpu_float, CV_32F, 1.0 / 255.0);
        cv::Mat host_float;
        gpu_float.download(host_float);
        extract_hwc_planes(host_float, host_data,
                           config_.input_height, config_.input_width,
                           config_.normalize_mean_std,
                           kMeanR, kMeanG, kMeanB, kStdR, kStdG, kStdB);
    } else {
        cv::Mat cpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);
        cv::Mat resized;
        cv::resize(cpu_image, resized,
                   cv::Size(config_.input_width, config_.input_height),
                   0, 0, cv::INTER_LINEAR);
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
        extract_hwc_planes(float_img, host_data,
                           config_.input_height, config_.input_width,
                           config_.normalize_mean_std,
                           kMeanR, kMeanG, kMeanB, kStdR, kStdG, kStdB);
    }

    cudaMemcpy(input_tensor.data, host_data.data(),
               host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void ClassifierNode::preprocess_crops(Frame& frame, Tensor& batch_tensor,
                                      std::vector<int>& valid_det_indices) {
    int orig_width = 0, orig_height = 0;
    get_image_dims(frame, orig_width, orig_height);
    if (orig_width <= 0 || orig_height <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    valid_det_indices.clear();

    int candidates = 0;
    for (size_t i = 0; i < frame.detections.size() && candidates < config_.max_batch_size; ++i) {
        if (matches_target_class(frame.detections[i].class_id)) {
            ++candidates;
        }
    }

    if (candidates == 0) {
        return;
    }

    batch_tensor = Tensor({candidates, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    std::vector<float> all_crops_data;
    all_crops_data.reserve(candidates * 3 * config_.input_height * config_.input_width);

    for (size_t i = 0; i < frame.detections.size(); ++i) {
        if (static_cast<int>(valid_det_indices.size()) >= config_.max_batch_size) break;
        if (!matches_target_class(frame.detections[i].class_id)) continue;

        std::vector<float> crop_data;
        if (crop_and_preprocess(frame, frame.detections[i], crop_data)) {
            valid_det_indices.push_back(static_cast<int>(i));
            all_crops_data.insert(all_crops_data.end(),
                                  crop_data.begin(), crop_data.end());
        }
    }

    if (valid_det_indices.empty()) {
        batch_tensor = Tensor();
        return;
    }

    int actual_batch = static_cast<int>(valid_det_indices.size());
    if (actual_batch != candidates) {
        batch_tensor = Tensor({actual_batch, 3, config_.input_height, config_.input_width},
                              DataType::FLOAT32, get_cuda_allocator());
    }

    cudaMemcpy(batch_tensor.data, all_crops_data.data(),
               all_crops_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

bool ClassifierNode::crop_and_preprocess(Frame& frame, const Detection& det,
                                         std::vector<float>& crop_data) {
    int orig_width = 0, orig_height = 0;
    get_image_dims(frame, orig_width, orig_height);

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
        cv::cuda::GpuMat gpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);
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
        extract_hwc_planes(host_float, crop_data,
                           config_.input_height, config_.input_width,
                           config_.normalize_mean_std,
                           kMeanR, kMeanG, kMeanB, kStdR, kStdG, kStdB);
    } else {
        cv::Mat cpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);
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
        extract_hwc_planes(float_img, crop_data,
                           config_.input_height, config_.input_width,
                           config_.normalize_mean_std,
                           kMeanR, kMeanG, kMeanB, kStdR, kStdG, kStdB);
    }

    return true;
}

void ClassifierNode::postprocess_whole_image(Frame& frame, const Tensor& output) {
    if (output.data == nullptr) return;

    int num_classes = 1;
    if (output.shape.size() >= 2) {
        num_classes = static_cast<int>(output.shape[1]);
    } else if (output.shape.size() == 1) {
        num_classes = static_cast<int>(output.shape[0]);
    }

    std::vector<float> host_output(output.nbytes / sizeof(float));
    cudaMemcpy(host_output.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);

    float prob = 0.0f;
    int cls = argmax_softmax(host_output.data(), num_classes, prob);

    Classification result;
    result.detection_index = -1;
    result.class_id = cls;
    result.confidence = prob;
    frame.classifications.push_back(result);
}

void ClassifierNode::postprocess_crops(Frame& frame, const Tensor& output,
                                       const std::vector<int>& valid_det_indices) {
    if (output.data == nullptr || valid_det_indices.empty()) return;

    int batch_size = static_cast<int>(valid_det_indices.size());
    int num_classes = 1;
    if (output.shape.size() >= 2) {
        num_classes = static_cast<int>(output.shape[1]);
    } else if (output.shape.size() == 1) {
        num_classes = static_cast<int>(output.shape[0]) / batch_size;
    }

    std::vector<float> host_output(output.nbytes / sizeof(float));
    cudaMemcpy(host_output.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        int det_idx = valid_det_indices[i];
        if (det_idx >= static_cast<int>(frame.detections.size())) continue;

        float prob = 0.0f;
        int cls = argmax_softmax(&host_output[i * num_classes], num_classes, prob);

        Classification result;
        result.detection_index = det_idx;
        result.class_id = cls;
        result.confidence = prob;
        frame.classifications.push_back(result);
    }
}

}  // namespace visionpipe
