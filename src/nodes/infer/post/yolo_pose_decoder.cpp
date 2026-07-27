#include "yolo_pose_decoder.h"

#include <algorithm>
#include <cmath>

#include <cuda_runtime_api.h>

namespace visionpipe {

namespace {

float bbox_iou(const float a[4], const float b[4]) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);

    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

void YoloPoseDecoder::decode(const Tensor& output,
                             std::vector<Detection>& detections,
                             std::vector<PoseResult>& poses,
                             const YoloPoseParams& params,
                             const LetterboxParams& letterbox_params,
                             int orig_width, int orig_height) {
    detections.clear();
    poses.clear();

    if (!output.valid() || output.shape.size() != 3) {
        return;
    }

    const int channels = static_cast<int>(output.shape[1]);
    const int num_anchors = static_cast<int>(output.shape[2]);
    const int expected = 4 + 1 + params.num_keypoints * 3;
    if (channels != expected || num_anchors <= 0) {
        return;
    }

    std::vector<float> host(output.numel());
    if (output.memory_type() == MemoryType::CUDA_DEVICE) {
        cudaMemcpy(host.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::copy_n(static_cast<const float*>(output.data), output.numel(), host.begin());
    }

    // 候选收集（letterbox 空间）
    struct Candidate {
        float bbox[4];
        float conf;
        std::vector<Keypoint> kpts;
    };
    std::vector<Candidate> cands;

    for (int i = 0; i < num_anchors; ++i) {
        const float conf = host[4 * num_anchors + i];
        if (conf < params.score_threshold) {
            continue;
        }

        Candidate c;
        const float cx = host[0 * num_anchors + i];
        const float cy = host[1 * num_anchors + i];
        const float w = host[2 * num_anchors + i];
        const float h = host[3 * num_anchors + i];
        c.bbox[0] = cx - 0.5f * w;
        c.bbox[1] = cy - 0.5f * h;
        c.bbox[2] = cx + 0.5f * w;
        c.bbox[3] = cy + 0.5f * h;
        c.conf = conf;

        c.kpts.resize(params.num_keypoints);
        for (int k = 0; k < params.num_keypoints; ++k) {
            c.kpts[k].x = host[(5 + k * 3 + 0) * num_anchors + i];
            c.kpts[k].y = host[(5 + k * 3 + 1) * num_anchors + i];
            c.kpts[k].score = host[(5 + k * 3 + 2) * num_anchors + i];
        }
        cands.push_back(std::move(c));
    }

    if (cands.empty()) {
        return;
    }

    // NMS（单类别）
    std::vector<int> order(cands.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&cands](int a, int b) {
        return cands[a].conf > cands[b].conf;
    });

    std::vector<bool> suppressed(cands.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (suppressed[order[i]]) continue;
        for (size_t j = i + 1; j < order.size(); ++j) {
            if (suppressed[order[j]]) continue;
            if (bbox_iou(cands[order[i]].bbox, cands[order[j]].bbox) > params.nms_threshold) {
                suppressed[order[j]] = true;
            }
        }
    }

    // letterbox 逆映射 → 归一化输出
    for (size_t i = 0; i < order.size() &&
                       static_cast<int>(detections.size()) < params.max_detections; ++i) {
        const int idx = order[i];
        if (suppressed[idx]) continue;
        Candidate& c = cands[idx];

        LetterboxResize::map_bbox_back(c.bbox, letterbox_params);
        c.bbox[0] = std::clamp(c.bbox[0], 0.0f, static_cast<float>(orig_width));
        c.bbox[1] = std::clamp(c.bbox[1], 0.0f, static_cast<float>(orig_height));
        c.bbox[2] = std::clamp(c.bbox[2], 0.0f, static_cast<float>(orig_width));
        c.bbox[3] = std::clamp(c.bbox[3], 0.0f, static_cast<float>(orig_height));

        Detection det;
        det.bbox[0] = c.bbox[0] / orig_width;
        det.bbox[1] = c.bbox[1] / orig_height;
        det.bbox[2] = c.bbox[2] / orig_width;
        det.bbox[3] = c.bbox[3] / orig_height;
        det.class_id = 0;  // person
        det.confidence = c.conf;
        det.track_id = -1;

        PoseResult pose;
        pose.detection_index = static_cast<int>(detections.size());
        pose.keypoints.resize(c.kpts.size());
        for (size_t k = 0; k < c.kpts.size(); ++k) {
            const float px = (c.kpts[k].x - letterbox_params.pad_x) / letterbox_params.scale;
            const float py = (c.kpts[k].y - letterbox_params.pad_y) / letterbox_params.scale;
            pose.keypoints[k].x = std::clamp(px / orig_width, 0.0f, 1.0f);
            pose.keypoints[k].y = std::clamp(py / orig_height, 0.0f, 1.0f);
            pose.keypoints[k].score = c.kpts[k].score;
        }

        detections.push_back(det);
        poses.push_back(std::move(pose));
    }
}

}  // namespace visionpipe
