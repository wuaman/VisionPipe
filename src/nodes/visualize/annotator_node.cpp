#include "annotator_node.h"

#include <cstring>

#include <opencv2/imgproc.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/error.h"
#include "core/logger.h"

namespace visionpipe {

namespace {

// 20 色色盘（BGR）
static const cv::Scalar kPalette[20] = {
    {56,  56,  255}, {151, 157, 255}, {31,  112, 255}, {29,  178, 255},
    {49,  210, 207}, {10,  249, 72},  {23,  204, 146}, {134, 219, 61},
    {52,  147, 26},  {187, 212, 0},   {168, 153, 44},  {255, 194, 0},
    {147, 69,  52},  {255, 115, 100}, {236, 24,  0},   {255, 56,  132},
    {133, 0,   82},  {255, 56,  203}, {200, 149, 255}, {199, 55,  255},
};

}  // namespace

AnnotatorNode::AnnotatorNode(const AnnotatorConfig& config, const std::string& name)
    : NodeBase(name)
    , config_(config) {}

void AnnotatorNode::process(Frame& frame) {
    if (!frame.has_image()) {
        return;
    }

    cv::Mat bgr = to_cpu_bgr(frame);

    if (config_.draw_masks && !frame.masks.empty()) {
        draw_masks_overlay(bgr, frame);
    }
    if (config_.draw_detections && !frame.detections.empty()) {
        draw_detections_overlay(bgr, frame);
    }
    if (config_.draw_tracks && !frame.tracks.empty()) {
        draw_tracks_overlay(bgr, frame);
    }
    if (config_.draw_keypoints && !frame.poses.empty()) {
        draw_keypoints_overlay(bgr, frame);
    }

    write_back(frame, bgr);
}

cv::Mat AnnotatorNode::to_cpu_bgr(const Frame& frame) {
    const auto& img = frame.image;

    int h = 0, w = 0, c = 1;
    if (img.shape.size() == 3 && img.shape[2] <= 4) {
        h = static_cast<int>(img.shape[0]);
        w = static_cast<int>(img.shape[1]);
        c = static_cast<int>(img.shape[2]);
    } else if (img.shape.size() == 3 && img.shape[0] <= 4) {
        c = static_cast<int>(img.shape[0]);
        h = static_cast<int>(img.shape[1]);
        w = static_cast<int>(img.shape[2]);
    } else if (img.shape.size() == 2) {
        h = static_cast<int>(img.shape[0]);
        w = static_cast<int>(img.shape[1]);
    }

    if (h <= 0 || w <= 0) {
        throw VisionPipeError("AnnotatorNode: invalid image shape");
    }

    int cv_type = (c == 3) ? CV_8UC3 : (c == 4 ? CV_8UC4 : CV_8UC1);
    cv::Mat bgr;

#ifdef VISIONPIPE_USE_CUDA
    if (img.memory_type() == MemoryType::CUDA_DEVICE) {
        cv::Mat cpu(h, w, cv_type);
        cudaError_t err = cudaMemcpy(cpu.data, img.data, img.nbytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw VisionPipeError(std::string("AnnotatorNode: cudaMemcpy failed: ") +
                                  cudaGetErrorString(err));
        }
        // Frames from source nodes are stored as RGB → convert to BGR for drawing
        if (c == 3) {
            cv::cvtColor(cpu, bgr, cv::COLOR_RGB2BGR);
        } else {
            bgr = cpu;
        }
    } else
#endif
    {
        cv::Mat src(h, w, cv_type, img.data);
        if (c == 3) {
            cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
        } else {
            bgr = src.clone();
        }
    }

    return bgr;
}

void AnnotatorNode::draw_masks_overlay(cv::Mat& bgr, const Frame& frame) {
    const int H = bgr.rows;
    const int W = bgr.cols;

    cv::Mat colored(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

    for (size_t i = 0; i < frame.masks.size(); ++i) {
        const auto& mask_data = frame.masks[i];
        if (mask_data.empty()) continue;

        // mask may be smaller than image if decode kept original dims
        int mh = H, mw = W;
        if (static_cast<int>(mask_data.size()) == H * W) {
            mh = H; mw = W;
        } else {
            // skip malformed masks
            continue;
        }

        cv::Scalar col = color_for(static_cast<int>(i));

        for (int y = 0; y < mh; ++y) {
            for (int x = 0; x < mw; ++x) {
                if (mask_data[y * mw + x]) {
                    auto& px = colored.at<cv::Vec3b>(y, x);
                    px[0] = static_cast<uint8_t>(col[0]);
                    px[1] = static_cast<uint8_t>(col[1]);
                    px[2] = static_cast<uint8_t>(col[2]);
                }
            }
        }
    }

    cv::addWeighted(bgr, 1.0 - config_.mask_alpha, colored, config_.mask_alpha, 0.0, bgr);
}

void AnnotatorNode::draw_keypoints_overlay(cv::Mat& bgr, const Frame& frame) {
    const int H = bgr.rows;
    const int W = bgr.cols;

    // COCO-17 骨架连线（关键点下标对）
    static const int kSkeleton[][2] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12},  // 腿
        {11, 12},                                // 髋
        {5, 11}, {6, 12},                        // 躯干
        {5, 6},                                  // 肩
        {5, 7}, {7, 9}, {6, 8}, {8, 10},         // 臂
        {0, 1}, {0, 2}, {1, 3}, {2, 4},          // 头
        {3, 5}, {4, 6},                          // 耳-肩
    };

    for (size_t p = 0; p < frame.poses.size(); ++p) {
        const auto& kpts = frame.poses[p].keypoints;
        const cv::Scalar col = color_for(static_cast<int>(p));

        // 骨架连线（仅 COCO-17 布局时绘制）
        if (kpts.size() >= 17) {
            for (const auto& edge : kSkeleton) {
                const Keypoint& a = kpts[edge[0]];
                const Keypoint& b = kpts[edge[1]];
                if (a.score < config_.kpt_score_threshold ||
                    b.score < config_.kpt_score_threshold) {
                    continue;
                }
                cv::line(bgr,
                         cv::Point(static_cast<int>(a.x * W), static_cast<int>(a.y * H)),
                         cv::Point(static_cast<int>(b.x * W), static_cast<int>(b.y * H)),
                         col, 2, cv::LINE_AA);
            }
        }

        for (const auto& kp : kpts) {
            if (kp.score < config_.kpt_score_threshold) {
                continue;
            }
            cv::circle(bgr,
                       cv::Point(static_cast<int>(kp.x * W), static_cast<int>(kp.y * H)),
                       3, col, -1, cv::LINE_AA);
        }
    }
}

void AnnotatorNode::draw_detections_overlay(cv::Mat& bgr, const Frame& frame) {
    const int H = bgr.rows;
    const int W = bgr.cols;

    for (size_t i = 0; i < frame.detections.size(); ++i) {
        const auto& det = frame.detections[i];

        int x1 = static_cast<int>(det.bbox[0] * W);
        int y1 = static_cast<int>(det.bbox[1] * H);
        int x2 = static_cast<int>(det.bbox[2] * W);
        int y2 = static_cast<int>(det.bbox[3] * H);

        cv::Scalar col = color_for(det.class_id);
        cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), col, 1);

        std::string label = class_name(det.class_id) +
                            " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        int ty = std::max(y1 - 4, ts.height);
        cv::rectangle(bgr, cv::Point(x1, ty - ts.height - 2),
                      cv::Point(x1 + ts.width, ty + baseline), col, cv::FILLED);
        cv::putText(bgr, label, cv::Point(x1, ty),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void AnnotatorNode::draw_tracks_overlay(cv::Mat& bgr, const Frame& frame) {
    const int H = bgr.rows;
    const int W = bgr.cols;

    for (const auto& trk : frame.tracks) {
        int x1 = static_cast<int>(trk.bbox[0] * W);
        int y1 = static_cast<int>(trk.bbox[1] * H);
        int x2 = static_cast<int>(trk.bbox[2] * W);
        int y2 = static_cast<int>(trk.bbox[3] * H);

        int idx = static_cast<int>(trk.track_id % 20);
        cv::Scalar col = color_for(idx);
        cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), col, 2);

        std::string label = "ID:" + std::to_string(trk.track_id);
        cv::putText(bgr, label, cv::Point(x1 + 2, y1 + 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
    }
}

void AnnotatorNode::write_back(Frame& frame, const cv::Mat& bgr) {
    const int H = bgr.rows;
    const int W = bgr.cols;
    size_t nbytes = static_cast<size_t>(H) * W * 3;

    Tensor cpu_tensor({static_cast<int64_t>(H), static_cast<int64_t>(W), 3LL},
                      DataType::UINT8, &cpu_alloc_);
    std::memcpy(cpu_tensor.data, bgr.data, nbytes);
    frame.image = std::move(cpu_tensor);
}

std::string AnnotatorNode::class_name(int id) const {
    if (!config_.class_names.empty() && id >= 0 &&
        id < static_cast<int>(config_.class_names.size())) {
        return config_.class_names[id];
    }
    return std::to_string(id);
}

cv::Scalar AnnotatorNode::color_for(int idx) {
    return kPalette[static_cast<size_t>(idx < 0 ? 0 : idx) % 20];
}

}  // namespace visionpipe
