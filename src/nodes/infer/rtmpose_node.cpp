#include "rtmpose_node.h"

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
#include "nodes/infer/post/simcc_decoder.h"

namespace visionpipe {

namespace {

// RTMPose/mmpose 部署常数（0-255 空间: mean 123.675/116.28/103.53, std 58.395/57.12/57.375）
// 这里在 [0,1] 空间等价表示
constexpr float kMeanR = 123.675f / 255.0f;
constexpr float kMeanG = 116.28f / 255.0f;
constexpr float kMeanB = 103.53f / 255.0f;
constexpr float kStdR = 58.395f / 255.0f;
constexpr float kStdG = 57.12f / 255.0f;
constexpr float kStdB = 57.375f / 255.0f;

std::shared_ptr<CudaAllocator> g_cuda_allocator;

CudaAllocator* get_cuda_allocator() {
    if (!g_cuda_allocator) {
        g_cuda_allocator = std::make_shared<CudaAllocator>();
    }
    return g_cuda_allocator.get();
}

}  // namespace

RtmPoseNode::RtmPoseNode(std::shared_ptr<IModelEngine> engine,
                         const RtmPoseConfig& config,
                         const std::string& name)
    : InferNode(std::move(engine), config.workers, 1, std::chrono::milliseconds(5), name)
    , config_(config) {}

RtmPoseNode::RtmPoseNode(std::shared_ptr<IModelEngine> engine,
                         const std::string& name)
    : RtmPoseNode(std::move(engine), RtmPoseConfig(), name) {}

bool RtmPoseNode::set_param(const std::string& name, const ParamValue& value) {
    std::lock_guard<std::mutex> lock(params_mutex_);

    if (name == "score_threshold") {
        if (std::holds_alternative<float>(value)) {
            config_.score_threshold = std::get<float>(value);
            return true;
        }
        if (std::holds_alternative<double>(value)) {
            config_.score_threshold = static_cast<float>(std::get<double>(value));
            return true;
        }
    } else if (name == "bbox_padding") {
        if (std::holds_alternative<float>(value)) {
            config_.bbox_padding = std::get<float>(value);
            return true;
        }
        if (std::holds_alternative<double>(value)) {
            config_.bbox_padding = static_cast<float>(std::get<double>(value));
            return true;
        }
    } else if (name == "max_batch_size") {
        if (std::holds_alternative<int>(value)) {
            config_.max_batch_size = std::get<int>(value);
            return true;
        }
    }

    return false;
}

bool RtmPoseNode::matches_target_class(int class_id) const {
    if (config_.target_classes.empty()) {
        return true;
    }
    return std::find(config_.target_classes.begin(),
                     config_.target_classes.end(),
                     class_id) != config_.target_classes.end();
}

void RtmPoseNode::get_image_dims(const Frame& frame, int& width, int& height) const {
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

void RtmPoseNode::process_batch(std::vector<Frame>& frames) {
    for (auto& frame : frames) {
        infer_frame(frame);
    }
}

RtmPoseNode::CropRect RtmPoseNode::compute_crop_rect(const Detection& det,
                                                     int orig_w, int orig_h) const {
    const float cx = 0.5f * (det.bbox[0] + det.bbox[2]) * orig_w;
    const float cy = 0.5f * (det.bbox[1] + det.bbox[3]) * orig_h;
    float w = (det.bbox[2] - det.bbox[0]) * orig_w;
    float h = (det.bbox[3] - det.bbox[1]) * orig_h;

    // 对齐模型输入纵横比（mmpose TopdownAffine 语义）
    const float aspect = static_cast<float>(config_.input_width) /
                         static_cast<float>(config_.input_height);
    if (w > aspect * h) {
        h = w / aspect;
    } else {
        w = aspect * h;
    }

    w *= config_.bbox_padding;
    h *= config_.bbox_padding;

    return CropRect{cx - 0.5f * w, cy - 0.5f * h, w, h};
}

void RtmPoseNode::crop_and_preprocess(const Frame& frame, const CropRect& rect,
                                      int orig_w, int orig_h,
                                      std::vector<float>& crop_data) const {
    const int in_w = config_.input_width;
    const int in_h = config_.input_height;

    // rect（原图空间，可超出边界）→ 模型输入的仿射映射；越界区域填 0
    const double sx = static_cast<double>(in_w) / rect.w;
    const double sy = static_cast<double>(in_h) / rect.h;
    cv::Mat M = (cv::Mat_<double>(2, 3) << sx, 0.0, -rect.x1 * sx,
                                            0.0, sy, -rect.y1 * sy);

    cv::Mat host_float;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        cv::cuda::GpuMat gpu_image(orig_h, orig_w, CV_8UC3, frame.image.data);
        cv::cuda::GpuMat gpu_warped;
        cv::cuda::warpAffine(gpu_image, gpu_warped, M, cv::Size(in_w, in_h),
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        cv::cuda::GpuMat gpu_rgb;
        cv::cuda::cvtColor(gpu_warped, gpu_rgb, cv::COLOR_BGR2RGB);
        cv::cuda::GpuMat gpu_float;
        gpu_rgb.convertTo(gpu_float, CV_32F, 1.0 / 255.0);
        gpu_float.download(host_float);
    } else {
        cv::Mat cpu_image(orig_h, orig_w, CV_8UC3, frame.image.data);
        cv::Mat warped;
        cv::warpAffine(cpu_image, warped, M, cv::Size(in_w, in_h),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        cv::Mat rgb;
        cv::cvtColor(warped, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(host_float, CV_32F, 1.0 / 255.0);
    }

    crop_data.resize(3 * static_cast<size_t>(in_h) * in_w);
    const int plane_size = in_h * in_w;
    for (int y = 0; y < in_h; ++y) {
        for (int x = 0; x < in_w; ++x) {
            const cv::Vec3f& px = host_float.at<cv::Vec3f>(y, x);
            crop_data[0 * plane_size + y * in_w + x] = (px[0] - kMeanR) / kStdR;
            crop_data[1 * plane_size + y * in_w + x] = (px[1] - kMeanG) / kStdG;
            crop_data[2 * plane_size + y * in_w + x] = (px[2] - kMeanB) / kStdB;
        }
    }
}

void RtmPoseNode::infer_frame(Frame& frame) {
    RtmPoseConfig config;
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        config = config_;
    }

    if (frame.detections.empty()) {
        return;
    }
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    int orig_w = 0, orig_h = 0;
    get_image_dims(frame, orig_w, orig_h);
    if (orig_w <= 0 || orig_h <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    // 收集目标框并构建裁剪 batch
    std::vector<int> det_indices;
    std::vector<CropRect> rects;
    std::vector<float> batch_data;
    const size_t crop_elems = 3 * static_cast<size_t>(config.input_height) * config.input_width;

    std::vector<float> crop_data;
    for (size_t i = 0; i < frame.detections.size(); ++i) {
        if (static_cast<int>(det_indices.size()) >= config.max_batch_size) break;
        if (!matches_target_class(frame.detections[i].class_id)) continue;

        CropRect rect = compute_crop_rect(frame.detections[i], orig_w, orig_h);
        if (rect.w <= 1.0f || rect.h <= 1.0f) continue;

        crop_and_preprocess(frame, rect, orig_w, orig_h, crop_data);
        batch_data.insert(batch_data.end(), crop_data.begin(), crop_data.end());
        det_indices.push_back(static_cast<int>(i));
        rects.push_back(rect);
    }

    if (det_indices.empty()) {
        return;
    }

    const int batch = static_cast<int>(det_indices.size());
    Tensor batch_tensor({batch, 3, config.input_height, config.input_width},
                        DataType::FLOAT32, get_cuda_allocator());
    cudaMemcpy(batch_tensor.data, batch_data.data(),
               batch * crop_elems * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<Tensor> outputs;
    run_inference_multi(batch_tensor, outputs);
    if (outputs.size() < 2) {
        throw InferError("RtmPoseNode expects 2 outputs (simcc_x, simcc_y) from engine");
    }

    // simcc_x: [batch, K, x_bins], simcc_y: [batch, K, y_bins]
    const Tensor& simcc_x = outputs[0];
    const Tensor& simcc_y = outputs[1];
    if (simcc_x.shape.size() != 3 || simcc_y.shape.size() != 3) {
        throw InferError("RtmPoseNode: unexpected SimCC output rank");
    }

    const int num_kpts = static_cast<int>(simcc_x.shape[1]);
    const int x_bins = static_cast<int>(simcc_x.shape[2]);
    const int y_bins = static_cast<int>(simcc_y.shape[2]);

    std::vector<float> host_x(simcc_x.numel());
    std::vector<float> host_y(simcc_y.numel());
    if (simcc_x.memory_type() == MemoryType::CUDA_DEVICE) {
        cudaMemcpy(host_x.data(), simcc_x.data, simcc_x.nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_y.data(), simcc_y.data, simcc_y.nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::copy_n(static_cast<const float*>(simcc_x.data), simcc_x.numel(), host_x.begin());
        std::copy_n(static_cast<const float*>(simcc_y.data), simcc_y.numel(), host_y.begin());
    }

    // 引擎输出 batch 维可能大于实际 batch（固定 max shape），按实际 batch 解码
    for (int b = 0; b < batch; ++b) {
        std::vector<Keypoint> kpts;
        SimccDecoder::decode(host_x.data() + static_cast<size_t>(b) * num_kpts * x_bins,
                             host_y.data() + static_cast<size_t>(b) * num_kpts * y_bins,
                             num_kpts, x_bins, y_bins,
                             config.simcc_split_ratio, kpts);

        // 模型输入空间 → 原图空间 → 归一化
        const CropRect& rect = rects[b];
        for (auto& kp : kpts) {
            const float px = rect.x1 + kp.x * rect.w / config.input_width;
            const float py = rect.y1 + kp.y * rect.h / config.input_height;
            kp.x = std::clamp(px / orig_w, 0.0f, 1.0f);
            kp.y = std::clamp(py / orig_h, 0.0f, 1.0f);
        }

        PoseResult pose;
        pose.detection_index = det_indices[b];
        pose.keypoints = std::move(kpts);
        frame.poses.push_back(std::move(pose));
    }
}

}  // namespace visionpipe
