#include "nodes/sink/mjpeg_sink.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#endif

#include "core/logger.h"
#include "core/tensor.h"

namespace visionpipe {

MjpegSink::MjpegSink(const MjpegSinkConfig& config, const std::string& name)
    : NodeBase(name)
    , config_(config)
    , jpeg_queue_(std::make_shared<BoundedQueue<std::vector<uint8_t>>>(
          config.buffer_capacity, OverflowPolicy::DROP_OLDEST)) {}

void MjpegSink::process(Frame& frame) {
    if (!frame.has_image()) {
        return;
    }

    const auto& img = frame.image;
    if (img.shape.size() < 2) {
        VP_LOG_WARN("MjpegSink '{}': unexpected tensor shape, skipping frame {}", name(), frame.frame_id);
        return;
    }

    cv::Mat bgr;

#ifdef VISIONPIPE_USE_CUDA
    if (img.memory_type() == MemoryType::CUDA_DEVICE) {
        // HWC or CHW layout detection
        int h, w, c;
        if (img.shape.size() == 3 && img.shape[2] <= 4) {
            // HWC
            h = static_cast<int>(img.shape[0]);
            w = static_cast<int>(img.shape[1]);
            c = static_cast<int>(img.shape[2]);
        } else if (img.shape.size() == 3 && img.shape[0] <= 4) {
            // CHW — not typical for raw frames but handle defensively
            c = static_cast<int>(img.shape[0]);
            h = static_cast<int>(img.shape[1]);
            w = static_cast<int>(img.shape[2]);
        } else {
            VP_LOG_WARN("MjpegSink '{}': cannot determine HWC from shape, skipping frame {}", name(), frame.frame_id);
            return;
        }

        int cv_type = (c == 3) ? CV_8UC3 : (c == 1 ? CV_8UC1 : CV_8UC4);
        cv::Mat cpu(h, w, cv_type);
        cudaError_t err = cudaMemcpy(cpu.data, img.data, img.nbytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            VP_LOG_ERROR("MjpegSink '{}': cudaMemcpy failed: {}", name(), cudaGetErrorString(err));
            return;
        }

        // Assume stored as RGB → convert to BGR for imencode
        if (c == 3) {
            cv::cvtColor(cpu, bgr, cv::COLOR_RGB2BGR);
        } else {
            bgr = cpu;
        }
    } else
#endif
    {
        // CPU tensor (HWC, UINT8)
        int h = static_cast<int>(img.shape[0]);
        int w = static_cast<int>(img.shape[1]);
        int c = (img.shape.size() >= 3) ? static_cast<int>(img.shape[2]) : 1;
        int cv_type = (c == 3) ? CV_8UC3 : (c == 1 ? CV_8UC1 : CV_8UC4);
        cv::Mat src(h, w, cv_type, img.data);

        if (c == 3) {
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
        } else {
            bgr = src.clone();
        }
    }

    std::vector<uint8_t> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
    if (!cv::imencode(".jpg", bgr, buf, params)) {
        VP_LOG_ERROR("MjpegSink '{}': imencode failed for frame {}", name(), frame.frame_id);
        return;
    }

    jpeg_queue_->push(std::move(buf));
}

std::optional<std::vector<uint8_t>> MjpegSink::pop_jpeg(
    std::chrono::milliseconds timeout) {
    return jpeg_queue_->pop_for(timeout);
}

}  // namespace visionpipe
