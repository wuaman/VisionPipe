#include "detector_node.h"

#include <chrono>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

#include "core/error.h"
#include "core/logger.h"
#include "hal/nvidia/cuda_allocator.h"

namespace visionpipe {

namespace {

constexpr size_t kQueueCapacity = 1024;

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
    : NodeBase(name)
    , engine_(std::move(engine))
    , config_(config)
    , workers_(config.workers == 0 ? 1 : config.workers)
    , owned_input_queue_(std::make_shared<BoundedQueue<Frame>>(kQueueCapacity, OverflowPolicy::BLOCK)) {
    if (!engine_) {
        throw ConfigError("DetectorNode requires a valid engine");
    }
    input_queue_ = owned_input_queue_.get();
    create_output_queue(kQueueCapacity, OverflowPolicy::BLOCK);
}

DetectorNode::DetectorNode(std::shared_ptr<IModelEngine> engine,
                           const std::string& name)
    : DetectorNode(engine, DetectorConfig(), name) {}

DetectorNode::~DetectorNode() {
    stop(false);
    wait_stop();
}

void DetectorNode::start() {
    if (state_ == NodeState::RUNNING) {
        return;
    }

    if (!input_queue_) {
        throw ConfigError("DetectorNode requires an input queue");
    }

    // 创建推理上下文
    contexts_.clear();
    contexts_.reserve(workers_);
    for (size_t i = 0; i < workers_; ++i) {
        auto context = engine_->create_context();
        if (!context) {
            throw InferError("IModelEngine::create_context returned null");
        }
        contexts_.push_back(std::move(context));
    }

    // 重置统计
    processed_count_ = 0;
    error_count_ = 0;
    last_frame_time_ = 0;
    frames_since_last_fps_ = 0;
    current_fps_ = 0.0;

    // 重置帧重排序
    {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        pending_outputs_.clear();
        next_output_frame_id_ = 0;
        next_output_initialized_ = false;
    }
    in_flight_frames_ = 0;

    state_ = NodeState::RUNNING;
    on_init();

    // 启动 worker 线程
    worker_threads_.clear();
    worker_threads_.reserve(workers_);
    for (size_t i = 0; i < workers_; ++i) {
        worker_threads_.emplace_back(&DetectorNode::worker_loop, this, i);
    }

    VP_LOG_INFO("DetectorNode '{}' started with {} worker(s)", name_, workers_);
}

void DetectorNode::stop(bool drain) {
    NodeState expected = NodeState::RUNNING;
    if (!state_.compare_exchange_strong(expected, NodeState::DRAINING)) {
        if (state_ == NodeState::INIT || state_ == NodeState::STOPPED) {
            return;
        }
    }

    if (!drain) {
        state_ = NodeState::STOPPED;
        if (input_queue_) {
            input_queue_->stop();
        }
        if (output_queue_) {
            output_queue_->stop();
        }
    }

    on_stop();
}

void DetectorNode::wait_stop() {
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();
    contexts_.clear();
}

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
                // ROI 参数格式：[x1, y1, x2, y1, ..., xn, yn]
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

void DetectorNode::worker_loop(size_t worker_index) {
    auto& context = contexts_.at(worker_index);

    while (!should_worker_exit()) {
        auto frame_opt = input_queue_->pop_for(std::chrono::milliseconds(100));
        if (!frame_opt.has_value()) {
            if (state_ == NodeState::DRAINING && input_queue_->empty()) {
                break;
            }
            continue;
        }

        Frame frame = std::move(*frame_opt);
        in_flight_frames_.fetch_add(1, std::memory_order_relaxed);

        // 记录 frame_id 用于重排序
        {
            std::lock_guard<std::mutex> lock(reorder_mutex_);
            if (!next_output_initialized_) {
                next_output_frame_id_ = frame.frame_id;
                next_output_initialized_ = true;
            }
        }

        try {
            // 预处理
            Tensor input_tensor;
            auto letterbox_params = preprocess(frame, input_tensor);

            // 保存原始图像尺寸
            int orig_width = frame.image.shape.size() >= 2
                ? static_cast<int>(frame.image.shape[1])
                : 640;
            int orig_height = frame.image.shape.size() >= 2
                ? static_cast<int>(frame.image.shape[0])
                : 640;

            // 推理
            Tensor output;
            context->infer(input_tensor, output);

            // 后处理
            postprocess(frame, output, letterbox_params, orig_width, orig_height);

            ++processed_count_;

            // 更新 FPS
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            std::lock_guard<std::mutex> fps_lock(fps_mutex_);
            ++frames_since_last_fps_;
            if (last_frame_time_ == 0) {
                last_frame_time_ = now;
            }
            if (frames_since_last_fps_ >= 10) {
                const int64_t window_start = last_frame_time_.exchange(now);
                if (window_start > 0) {
                    const double elapsed_sec = static_cast<double>(now - window_start) / 1e9;
                    if (elapsed_sec > 0.0) {
                        current_fps_ = frames_since_last_fps_.load() / elapsed_sec;
                    }
                }
                frames_since_last_fps_ = 0;
            }
        } catch (const std::exception& e) {
            ++error_count_;
            in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
            on_error(frame, e.what());
            continue;
        }

        // 加入重排序队列
        {
            std::lock_guard<std::mutex> lock(reorder_mutex_);
            pending_outputs_.emplace(frame.frame_id, std::move(frame));
            emit_ready_frames_locked();
        }
        in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
    }

    // 最后发射剩余帧
    {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        emit_ready_frames_locked();
        if (state_ == NodeState::DRAINING && input_queue_ && input_queue_->empty() &&
            pending_outputs_.empty() && in_flight_frames_.load(std::memory_order_relaxed) == 0) {
            state_ = NodeState::STOPPED;
            if (output_queue_) {
                output_queue_->stop();
            }
        }
    }
}

LetterboxParams DetectorNode::preprocess(Frame& frame, Tensor& input_tensor) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    // 获取原始图像尺寸
    int orig_width = 0;
    int orig_height = 0;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        // GPU 图像
        if (frame.image.shape.size() == 3) {
            if (frame.image.shape[2] == 3) {
                // HWC 格式
                orig_height = static_cast<int>(frame.image.shape[0]);
                orig_width = static_cast<int>(frame.image.shape[1]);
            } else if (frame.image.shape[0] == 3) {
                // CHW 格式
                orig_height = static_cast<int>(frame.image.shape[1]);
                orig_width = static_cast<int>(frame.image.shape[2]);
            }
        } else if (frame.image.shape.size() == 2) {
            // HW 格式（灰度图）
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    } else {
        // CPU 图像
        if (frame.image.shape.size() == 3) {
            orig_height = static_cast<int>(frame.image.shape[0]);
            orig_width = static_cast<int>(frame.image.shape[1]);
        }
    }

    if (orig_width <= 0 || orig_height <= 0) {
        throw InferError("Invalid image dimensions in frame");
    }

    // 计算 letterbox 参数
    auto letterbox_params = LetterboxResize::compute_params(
        orig_width, orig_height, config_.input_width, config_.input_height);

    // 分配输入 tensor
    input_tensor = Tensor({1, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
        // GPU 图像处理
        int cv_type = CV_8UC3;
        if (frame.image.shape.size() == 3 && frame.image.shape[2] == 3) {
            cv_type = CV_8UC3;
        } else if (frame.image.dtype == DataType::UINT8) {
            cv_type = CV_8UC1;
        }

        cv::cuda::GpuMat gpu_image(orig_height, orig_width, cv_type, frame.image.data);

        // 使用 CUDA Stream 确保操作同步
        cv::cuda::Stream stream;

        // 执行 letterbox resize
        cv::cuda::GpuMat resized;
        LetterboxResize::compute_gpu(gpu_image, resized, letterbox_params, 0);

        // BGR -> RGB
        cv::cuda::GpuMat rgb;
        cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB, 0, stream);

        // 转换为 float 并归一化
        cv::cuda::GpuMat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0, 0.0, stream);

        // 等待 GPU 操作完成
        stream.waitForCompletion();

        // 下载到 CPU 进行 HWC -> CHW 转换
        cv::Mat host_float;
        float_img.download(host_float);

        // HWC -> CHW 并复制到 GPU tensor
        float* output_ptr = static_cast<float*>(input_tensor.data);
        const int plane_size = config_.input_height * config_.input_width;

        // 在 CPU 上转换 HWC -> CHW
        std::vector<float> host_chw(3 * plane_size);
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
                host_chw[0 * plane_size + h * config_.input_width + w] = pixel[0];  // R
                host_chw[1 * plane_size + h * config_.input_width + w] = pixel[1];  // G
                host_chw[2 * plane_size + h * config_.input_width + w] = pixel[2];  // B
            }
        }

        // 上传到 GPU
        cudaMemcpy(input_tensor.data, host_chw.data(), input_tensor.nbytes, cudaMemcpyHostToDevice);
    } else {
        // CPU 图像处理
        int cv_type = CV_8UC3;
        cv::Mat cpu_image(orig_height, orig_width, cv_type, frame.image.data);

        // 执行 letterbox resize
        cv::Mat resized;
        LetterboxResize::compute_cpu(cpu_image, resized, letterbox_params);

        // BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // 转换为 float 并归一化
        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

        // HWC -> CHW 并复制到 GPU
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
    NmsParams nms_params;
    nms_params.score_threshold = config_.score_threshold;
    nms_params.iou_threshold = config_.nms_threshold;
    nms_params.max_detections = config_.max_detections;

    DetectionDecoder::decode(output, frame.detections, nms_params,
                             letterbox_params, orig_width, orig_height);

    // 应用 ROI 过滤
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

    // 计算检测框中心点
    float cx = (det.bbox[0] + det.bbox[2]) / 2.0f;
    float cy = (det.bbox[1] + det.bbox[3]) / 2.0f;

    // 检查中心点是否在任一 ROI 多边形内
    for (const auto& polygon : roi_polygons_) {
        if (cv::pointPolygonTest(polygon, cv::Point2f(cx, cy), false) >= 0) {
            return true;
        }
    }

    return false;
}

bool DetectorNode::should_worker_exit() const {
    if (state_ == NodeState::STOPPED) {
        return true;
    }
    if (state_ == NodeState::DRAINING && input_queue_ && input_queue_->empty()) {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        return pending_outputs_.empty() && in_flight_frames_.load(std::memory_order_relaxed) == 0;
    }
    return false;
}

void DetectorNode::emit_ready_frames_locked() {
    while (next_output_initialized_) {
        auto it = pending_outputs_.find(next_output_frame_id_);
        if (it == pending_outputs_.end()) {
            break;
        }

        if (output_queue_) {
            output_queue_->push(std::move(it->second));
        }
        pending_outputs_.erase(it);
        ++next_output_frame_id_;
    }
}

void DetectorNode::process(Frame& frame) {
    // 同步处理单帧（用于简单场景）
    if (contexts_.empty()) {
        throw InferError("DetectorNode is not started");
    }

    Tensor input_tensor;
    auto letterbox_params = preprocess(frame, input_tensor);

    int orig_width = frame.image.shape.size() >= 2
        ? static_cast<int>(frame.image.shape[1])
        : 640;
    int orig_height = frame.image.shape.size() >= 2
        ? static_cast<int>(frame.image.shape[0])
        : 640;

    Tensor output;
    contexts_.front()->infer(input_tensor, output);

    postprocess(frame, output, letterbox_params, orig_width, orig_height);
}

}  // namespace visionpipe
