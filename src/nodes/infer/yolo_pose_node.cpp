#include "yolo_pose_node.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime_api.h>

#include "core/error.h"
#include "core/logger.h"
#include "hal/nvidia/cuda_allocator.h"
#include "nodes/infer/post/yolo_pose_decoder.h"

namespace visionpipe {

namespace {

std::shared_ptr<CudaAllocator> g_cuda_allocator;

CudaAllocator* get_cuda_allocator() {
    if (!g_cuda_allocator) {
        g_cuda_allocator = std::make_shared<CudaAllocator>();
    }
    return g_cuda_allocator.get();
}

}  // namespace

YoloPoseNode::YoloPoseNode(std::shared_ptr<IModelEngine> engine,
                           const YoloPoseConfig& config,
                           const std::string& name)
    : InferNode(std::move(engine), config.workers, 1, std::chrono::milliseconds(5), name)
    , config_(config) {}

YoloPoseNode::YoloPoseNode(std::shared_ptr<IModelEngine> engine,
                           const std::string& name)
    : YoloPoseNode(std::move(engine), YoloPoseConfig(), name) {}

bool YoloPoseNode::set_param(const std::string& name, const ParamValue& value) {
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
    } else if (name == "nms_threshold") {
        if (std::holds_alternative<float>(value)) {
            config_.nms_threshold = std::get<float>(value);
            return true;
        }
        if (std::holds_alternative<double>(value)) {
            config_.nms_threshold = static_cast<float>(std::get<double>(value));
            return true;
        }
    } else if (name == "max_detections") {
        if (std::holds_alternative<int>(value)) {
            config_.max_detections = std::get<int>(value);
            return true;
        }
    }

    return false;
}

void YoloPoseNode::process_batch(std::vector<Frame>& frames) {
    YoloPoseParams params;
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        params.score_threshold = config_.score_threshold;
        params.nms_threshold = config_.nms_threshold;
        params.max_detections = config_.max_detections;
        params.num_keypoints = config_.num_keypoints;
        params.input_width = config_.input_width;
        params.input_height = config_.input_height;
    }

    for (auto& frame : frames) {
        Tensor input_tensor;
        auto letterbox_params = preprocess(frame, input_tensor);

        int orig_width = 0, orig_height = 0;
        if (frame.image.shape.size() == 3) {
            if (frame.image.memory_type() == MemoryType::CUDA_DEVICE &&
                frame.image.shape[0] == 3) {
                orig_height = static_cast<int>(frame.image.shape[1]);
                orig_width = static_cast<int>(frame.image.shape[2]);
            } else {
                orig_height = static_cast<int>(frame.image.shape[0]);
                orig_width = static_cast<int>(frame.image.shape[1]);
            }
        }

        Tensor output;
        run_inference(input_tensor, output);

        YoloPoseDecoder::decode(output, frame.detections, frame.poses,
                                params, letterbox_params, orig_width, orig_height);
    }
}

LetterboxParams YoloPoseNode::preprocess(Frame& frame, Tensor& input_tensor) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    int orig_width = 0;
    int orig_height = 0;

    if (frame.image.shape.size() == 3) {
        if (frame.image.memory_type() == MemoryType::CUDA_DEVICE &&
            frame.image.shape[0] == 3) {
            orig_height = static_cast<int>(frame.image.shape[1]);
            orig_width = static_cast<int>(frame.image.shape[2]);
        } else {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    } else if (frame.image.shape.size() == 2) {
        orig_height = static_cast<int>(frame.image.shape[0]);
        orig_width = static_cast<int>(frame.image.shape[1]);
    }

    if (orig_width <= 0 || orig_height <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    auto letterbox_params = LetterboxResize::compute_params(
        orig_width, orig_height, config_.input_width, config_.input_height);

    input_tensor = Tensor({1, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    const int plane_size = config_.input_height * config_.input_width;
    std::vector<float> host_chw(3 * static_cast<size_t>(plane_size));
    cv::Mat host_float;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        cv::cuda::GpuMat gpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);

        cv::cuda::GpuMat resized;
        LetterboxResize::compute_gpu(gpu_image, resized, letterbox_params, 0);

        cv::cuda::GpuMat rgb;
        cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::cuda::GpuMat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
        float_img.download(host_float);
    } else {
        cv::Mat cpu_image(orig_height, orig_width, CV_8UC3, frame.image.data);

        cv::Mat resized;
        LetterboxResize::compute_cpu(cpu_image, resized, letterbox_params);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        rgb.convertTo(host_float, CV_32F, 1.0 / 255.0);
    }

    for (int h = 0; h < config_.input_height; ++h) {
        for (int w = 0; w < config_.input_width; ++w) {
            const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
            host_chw[0 * plane_size + h * config_.input_width + w] = pixel[0];
            host_chw[1 * plane_size + h * config_.input_width + w] = pixel[1];
            host_chw[2 * plane_size + h * config_.input_width + w] = pixel[2];
        }
    }

    cudaMemcpy(input_tensor.data, host_chw.data(), input_tensor.nbytes,
               cudaMemcpyHostToDevice);

    return letterbox_params;
}

}  // namespace visionpipe
