#include "classifier_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

#include <cuda_runtime_api.h>

#include "core/error.h"
#include "core/logger.h"
#include "hal/nvidia/cuda_allocator.h"

namespace visionpipe {

namespace {

constexpr size_t kQueueCapacity = 1024;

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
    : NodeBase(name)
    , engine_(std::move(engine))
    , config_(config)
    , workers_(config.workers == 0 ? 1 : config.workers)
    , owned_input_queue_(std::make_shared<BoundedQueue<Frame>>(kQueueCapacity, OverflowPolicy::BLOCK)) {
    if (!engine_) {
        throw ConfigError("ClassifierNode requires a valid engine");
    }
    input_queue_ = owned_input_queue_.get();
    create_output_queue(kQueueCapacity, OverflowPolicy::BLOCK);
}

ClassifierNode::ClassifierNode(std::shared_ptr<IModelEngine> engine,
                               const std::string& name)
    : ClassifierNode(engine, ClassifierConfig(), name) {}

ClassifierNode::~ClassifierNode() {
    stop(false);
    wait_stop();
}

void ClassifierNode::start() {
    if (state_ == NodeState::RUNNING) {
        return;
    }

    if (!input_queue_) {
        throw ConfigError("ClassifierNode requires an input queue");
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
        worker_threads_.emplace_back(&ClassifierNode::worker_loop, this, i);
    }

    VP_LOG_INFO("ClassifierNode '{}' started with {} worker(s), max_batch={}",
                name_, workers_, config_.max_batch_size);
}

void ClassifierNode::stop(bool drain) {
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

void ClassifierNode::wait_stop() {
    for (auto& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();
    contexts_.clear();
}

void ClassifierNode::worker_loop(size_t worker_index) {
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
            // 如果没有检测结果，直接透传
            if (frame.detections.empty()) {
                // 无需推理，直接输出
                {
                    std::lock_guard<std::mutex> lock(reorder_mutex_);
                    pending_outputs_.emplace(frame.frame_id, std::move(frame));
                    emit_ready_frames_locked();
                }
                in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }

            // 预处理：裁剪 crops 并打包成 batch
            Tensor batch_tensor;
            std::vector<int> valid_crop_indices;
            preprocess(frame, batch_tensor, valid_crop_indices);

            // 如果没有有效 crop，直接透传
            if (valid_crop_indices.empty()) {
                {
                    std::lock_guard<std::mutex> lock(reorder_mutex_);
                    pending_outputs_.emplace(frame.frame_id, std::move(frame));
                    emit_ready_frames_locked();
                }
                in_flight_frames_.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }

            // 推理
            Tensor output;
            context->infer(batch_tensor, output);

            // 后处理：softmax 并回写结果
            postprocess(frame, output, valid_crop_indices);

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

void ClassifierNode::preprocess(Frame& frame, Tensor& batch_tensor,
                                std::vector<int>& valid_crop_indices) {
    if (!frame.has_image()) {
        throw InferError("Frame has no image data");
    }

    valid_crop_indices.clear();

    // 获取原始图像尺寸
    int orig_width = 0;
    int orig_height = 0;

    if (frame.image.memory_type() == MemoryType::CUDA_DEVICE) {
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

    // 限制 batch 大小
    int batch_size = std::min(static_cast<int>(frame.detections.size()),
                              config_.max_batch_size);

    // 分配 batch tensor
    batch_tensor = Tensor({batch_size, 3, config_.input_height, config_.input_width},
                          DataType::FLOAT32, get_cuda_allocator());

    // 准备所有 crop 数据
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

    // 如果没有有效 crop，清空 batch_tensor
    if (valid_crop_indices.empty()) {
        batch_tensor = Tensor();
        return;
    }

    // 更新实际的 batch 大小
    int actual_batch = static_cast<int>(valid_crop_indices.size());
    if (actual_batch != batch_size) {
        // 重新分配正确大小的 tensor
        batch_tensor = Tensor({actual_batch, 3, config_.input_height, config_.input_width},
                              DataType::FLOAT32, get_cuda_allocator());
    }

    // 上传到 GPU
    cudaMemcpy(batch_tensor.data, all_crops_data.data(),
               all_crops_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

bool ClassifierNode::crop_and_preprocess(Frame& frame, const Detection& det,
                                         std::vector<float>& crop_data) {
    // 获取原始图像尺寸
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

    // 将归一化坐标转换为像素坐标
    int x1 = static_cast<int>(det.bbox[0] * orig_width);
    int y1 = static_cast<int>(det.bbox[1] * orig_height);
    int x2 = static_cast<int>(det.bbox[2] * orig_width);
    int y2 = static_cast<int>(det.bbox[3] * orig_height);

    // 裁剪区域边界检查
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
        // GPU 图像处理
        cv::cuda::GpuMat gpu_image(orig_height, orig_width, cv_type, frame.image.data);

        // 裁剪
        cv::Rect roi(x1, y1, crop_width, crop_height);
        cv::cuda::GpuMat gpu_crop = gpu_image(roi);

        // Resize 到模型输入尺寸
        cv::cuda::GpuMat gpu_resized;
        cv::cuda::resize(gpu_crop, gpu_resized,
                         cv::Size(config_.input_width, config_.input_height),
                         0, 0, cv::INTER_LINEAR);

        // BGR -> RGB
        cv::cuda::GpuMat gpu_rgb;
        cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);

        // 转换为 float 并归一化
        cv::cuda::GpuMat gpu_float;
        gpu_rgb.convertTo(gpu_float, CV_32F, 1.0 / 255.0);

        // 下载到 CPU
        cv::Mat host_float;
        gpu_float.download(host_float);

        // HWC -> CHW 并应用 mean/std 归一化
        const int plane_size = config_.input_height * config_.input_width;
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                const cv::Vec3f& pixel = host_float.at<cv::Vec3f>(h, w);
                float r = pixel[0];
                float g = pixel[1];
                float b = pixel[2];

                if (config_.normalize_mean_std) {
                    // ImageNet 归一化
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
        // CPU 图像处理
        cv::Mat cpu_image(orig_height, orig_width, cv_type, frame.image.data);

        // 裁剪
        cv::Rect roi(x1, y1, crop_width, crop_height);
        cv::Mat cpu_crop = cpu_image(roi);

        // Resize
        cv::Mat resized;
        cv::resize(cpu_crop, resized,
                   cv::Size(config_.input_width, config_.input_height),
                   0, 0, cv::INTER_LINEAR);

        // BGR -> RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // 转换为 float 并归一化
        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

        // HWC -> CHW 并应用 mean/std 归一化
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

    // 输出格式假设为 [batch_size, num_classes]
    int batch_size = static_cast<int>(valid_crop_indices.size());
    int num_classes = 1;
    if (output.shape.size() >= 2) {
        num_classes = static_cast<int>(output.shape[1]);
    } else if (output.shape.size() == 1) {
        // 单输出情况
        num_classes = static_cast<int>(output.shape[0]) / batch_size;
    }

    // 下载输出到 CPU
    std::vector<float> host_output(output.nbytes / sizeof(float));
    cudaMemcpy(host_output.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);

    // 对每个 crop 应用 softmax 并获取最大值
    for (int i = 0; i < batch_size; ++i) {
        int det_idx = valid_crop_indices[i];
        if (det_idx >= static_cast<int>(frame.detections.size())) {
            continue;
        }

        // 获取该 crop 的 logits
        std::vector<float> logits(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            logits[c] = host_output[i * num_classes + c];
        }

        // Softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            logits[c] = std::exp(logits[c] - max_logit);
            sum_exp += logits[c];
        }
        for (int c = 0; c < num_classes; ++c) {
            logits[c] /= sum_exp;
        }

        // 找到最大概率的类别
        int best_class = 0;
        float best_prob = logits[0];
        for (int c = 1; c < num_classes; ++c) {
            if (logits[c] > best_prob) {
                best_prob = logits[c];
                best_class = c;
            }
        }

        // 回写到 detection
        frame.detections[det_idx].class_id = best_class;
        frame.detections[det_idx].confidence = best_prob;
    }
}

bool ClassifierNode::should_worker_exit() const {
    if (state_ == NodeState::STOPPED) {
        return true;
    }
    if (state_ == NodeState::DRAINING && input_queue_ && input_queue_->empty()) {
        std::lock_guard<std::mutex> lock(reorder_mutex_);
        return pending_outputs_.empty() && in_flight_frames_.load(std::memory_order_relaxed) == 0;
    }
    return false;
}

void ClassifierNode::emit_ready_frames_locked() {
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

void ClassifierNode::process(Frame& frame) {
    // 同步处理单帧（用于简单场景）
    if (contexts_.empty()) {
        throw InferError("ClassifierNode is not started");
    }

    // 如果没有检测结果，直接返回
    if (frame.detections.empty()) {
        return;
    }

    Tensor batch_tensor;
    std::vector<int> valid_crop_indices;
    preprocess(frame, batch_tensor, valid_crop_indices);

    if (valid_crop_indices.empty()) {
        return;
    }

    Tensor output;
    contexts_.front()->infer(batch_tensor, output);

    postprocess(frame, output, valid_crop_indices);
}

}  // namespace visionpipe
