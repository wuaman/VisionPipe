#include "detector_node.h"

#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>

#include "core/error.h"
#include "core/logger.h"
#include "hal/nvidia/cuda_allocator.h"

namespace visionpipe {

namespace {

// 全局 CUDA allocator（用于预处理 tensor）
std::shared_ptr<CudaAllocator> g_cuda_allocator;

CudaAllocator* get_cuda_allocator() {
    if (!g_cuda_allocator) {
        g_cuda_allocator = std::make_shared<CudaAllocator>();
    }
    return g_cuda_allocator.get();
}

}  // namespace

DetectorNode::DetectorNode(std::shared_ptr<IModelEngine> engine,
                           const DetectorConfig& config,
                           const std::string& name)
    : InferNode(std::move(engine), config.workers, 1, std::chrono::milliseconds(5), name)
    , config_(config) {}

DetectorNode::DetectorNode(std::shared_ptr<IModelEngine> engine,
                           const std::string& name)
    : DetectorNode(std::move(engine), DetectorConfig(), name) {}

bool DetectorNode::set_param(const std::string& name, const ParamValue& value) {
    std::lock_guard<std::mutex> lock(params_mutex_);

    try {
        if (name == "score_threshold") {
            if (std::holds_alternative<float>(value)) {
                config_.score_threshold = std::get<float>(value);
                return true;
            } else if (std::holds_alternative<double>(value)) {
                config_.score_threshold = static_cast<float>(std::get<double>(value));
                return true;
            }
        } else if (name == "nms_threshold") {
            if (std::holds_alternative<float>(value)) {
                config_.nms_threshold = std::get<float>(value);
                return true;
            } else if (std::holds_alternative<double>(value)) {
                config_.nms_threshold = static_cast<float>(std::get<double>(value));
                return true;
            }
        } else if (name == "max_detections") {
            if (std::holds_alternative<int>(value)) {
                config_.max_detections = std::get<int>(value);
                return true;
            }
        } else if (name == "roi") {
            if (std::holds_alternative<std::vector<float>>(value)) {
                auto coords = std::get<std::vector<float>>(value);
                if (coords.size() >= 6 && coords.size() % 2 == 0) {
                    std::vector<cv::Point2f> polygon;
                    for (size_t i = 0; i < coords.size(); i += 2) {
                        polygon.emplace_back(coords[i], coords[i + 1]);
                    }
                    std::lock_guard<std::mutex> roi_lock(roi_mutex_);
                    roi_polygons_.clear();
                    roi_polygons_.push_back(std::move(polygon));
                    return true;
                }
            }
        }
    } catch (const std::exception& e) {
        VP_LOG_ERROR("DetectorNode '{}': failed to set param '{}': {}",
                     name_, name, e.what());
        return false;
    }

    return NodeBase::set_param(name, value);
}

void DetectorNode::set_roi(const std::vector<std::vector<float>>& polygons) {
    std::lock_guard<std::mutex> lock(roi_mutex_);
    roi_polygons_.clear();

    for (const auto& coords : polygons) {
        if (coords.size() >= 6 && coords.size() % 2 == 0) {
            std::vector<cv::Point2f> polygon;
            for (size_t i = 0; i < coords.size(); i += 2) {
                polygon.emplace_back(coords[i], coords[i + 1]);
            }
            roi_polygons_.push_back(std::move(polygon));
        }
    }
}

void DetectorNode::clear_roi() {
    std::lock_guard<std::mutex> lock(roi_mutex_);
    roi_polygons_.clear();
}

void DetectorNode::process_batch(std::vector<Frame>& frames) {
    for (auto& frame : frames) {
        Tensor input_tensor;
        auto letterbox_params = preprocess(frame, input_tensor);

        int orig_width = frame.image.shape.size() >= 2
            ? static_cast<int>(frame.image.shape[1]) : 640;
        int orig_height = frame.image.shape.size() >= 2
            ? static_cast<int>(frame.image.shape[0]) : 640;

        Tensor output;
        run_inference(input_tensor, output);

        postprocess(frame, output, letterbox_params, orig_width, orig_height);
    }
}

LetterboxParams DetectorNode::preprocess(Frame& frame, Tensor& input_tensor) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

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
        } else if (frame.image.shape.size() == 2) {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
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

    auto letterbox_params = LetterboxResize::compute_params(
        orig_width, orig_height, config_.input_width, config_.input_height);

    input_tensor = Tensor({1, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        int cv_type = CV_8UC3;
        if (frame.image.shape.size() == 3 && frame.image.shape[2] == 3) {
            cv_type = CV_8UC3;
        } else if (frame.image.dtype == DataType::UINT8) {
            cv_type = CV_8UC1;
        }

        cv::cuda::GpuMat gpu_image(orig_height, orig_width, cv_type, frame.image.data);
        cv::cuda::Stream stream;

        cv::cuda::GpuMat resized;
        LetterboxResize::compute_gpu(gpu_image, resized, letterbox_params, 0);

        cv::cuda::GpuMat rgb;
        cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB, 0, stream);

        cv::cuda::GpuMat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0, 0.0, stream);
        stream.waitForCompletion();

        cv::Mat host_float;
        float_img.download(host_float);

        const int plane_size = config_.input_height * config_.input_width;
        std::vector<float> host_chw(3 * plane_size);
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
                host_chw[0 * plane_size + h * config_.input_width + w] = pixel[0];
                host_chw[1 * plane_size + h * config_.input_width + w] = pixel[1];
                host_chw[2 * plane_size + h * config_.input_width + w] = pixel[2];
            }
        }

        cudaMemcpy(input_tensor.data, host_chw.data(), input_tensor.nbytes, cudaMemcpyHostToDevice);
    } else {
        int cv_type = CV_8UC3;
        cv::Mat cpu_image(orig_height, orig_width, cv_type, frame.image.data);

        cv::Mat resized;
        LetterboxResize::compute_cpu(cpu_image, resized, letterbox_params);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

        std::vector<float> host_data(3 * config_.input_height * config_.input_width);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < config_.input_height; ++h) {
                for (int w = 0; w < config_.input_width; ++w) {
                    host_data[c * config_.input_height * config_.input_width +
                              h * config_.input_width + w] =
                        float_img.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        cudaMemcpy(input_tensor.data, host_data.data(), input_tensor.nbytes,
                   cudaMemcpyHostToDevice);
    }

    return letterbox_params;
}

void DetectorNode::postprocess(Frame& frame, const Tensor& output,
                               const LetterboxParams& letterbox_params,
                               int orig_width, int orig_height) {
    DetectorConfig config;
    {
        std::lock_guard<std::mutex> lock(params_mutex_);
        config = config_;
    }

    NmsParams nms_params;
    nms_params.score_threshold = config.score_threshold;
    nms_params.iou_threshold = config.nms_threshold;
    nms_params.max_detections = config.max_detections;

    DetectionDecoder::decode(output, frame.detections, nms_params,
                             letterbox_params, orig_width, orig_height);

    std::lock_guard<std::mutex> lock(roi_mutex_);
    if (!roi_polygons_.empty()) {
        frame.detections.erase(
            std::remove_if(frame.detections.begin(), frame.detections.end(),
                          [this](const Detection& det) { return !is_in_roi(det); }),
            frame.detections.end());
    }
}

bool DetectorNode::is_in_roi(const Detection& det) const {
    if (roi_polygons_.empty()) {
        return true;
    }

    float cx = (det.bbox[0] + det.bbox[2]) / 2.0f;
    float cy = (det.bbox[1] + det.bbox[3]) / 2.0f;

    for (const auto& polygon : roi_polygons_) {
        if (cv::pointPolygonTest(polygon, cv::Point2f(cx, cy), false) >= 0) {
            return true;
        }
    }

    return false;
}

}  // namespace visionpipe
