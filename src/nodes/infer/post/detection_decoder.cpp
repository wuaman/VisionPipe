#include "detection_decoder.h"

#include <algorithm>
#include <cmath>

namespace visionpipe {

void DetectionDecoder::decode(const Tensor& output,
                               std::vector<Detection>& detections,
                               const NmsParams& nms_params,
                               const LetterboxParams& letterbox_params,
                               int orig_width, int orig_height) {
    detections.clear();
    if (!output.valid() || output.shape.size() != 3) {
        return;
    }

    // YOLOv8 输出格式：[1, 84, num_anchors]
    // 84 = 4 (bbox: cx, cy, w, h) + 80 (class scores)
    const int num_classes = static_cast<int>(output.shape[1]) - 4;
    const int num_anchors = static_cast<int>(output.shape[2]);

    if (num_classes <= 0 || num_anchors <= 0) {
        return;
    }

    // 将 GPU 数据下载到 CPU 进行后处理
    std::vector<float> host_data(output.numel());
    if (output.memory_type() == MemoryType::CUDA_DEVICE) {
        cudaMemcpy(host_data.data(), output.data, output.nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::copy_n(static_cast<const float*>(output.data), output.numel(), host_data.begin());
    }

    // 解析所有 anchor
    std::vector<Detection> all_detections;
    all_detections.reserve(num_anchors);

    for (int i = 0; i < num_anchors; ++i) {
        // 找到最大类别分数
        float max_score = 0.0f;
        int max_class = 0;

        for (int c = 0; c < num_classes; ++c) {
            float score = host_data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }

        // 过滤低置信度检测
        if (max_score < nms_params.score_threshold) {
            continue;
        }

        // 提取 bbox（中心坐标格式）
        float cx = host_data[0 * num_anchors + i];
        float cy = host_data[1 * num_anchors + i];
        float w = host_data[2 * num_anchors + i];
        float h = host_data[3 * num_anchors + i];

        // 转换为角点坐标
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        Detection det;
        det.bbox[0] = x1;
        det.bbox[1] = y1;
        det.bbox[2] = x2;
        det.bbox[3] = y2;
        det.class_id = max_class;
        det.confidence = max_score;
        det.track_id = -1;

        all_detections.push_back(det);
    }

    // 执行 NMS
    nms(all_detections, nms_params.iou_threshold);

    // 限制检测数量
    if (static_cast<int>(all_detections.size()) > nms_params.max_detections) {
        all_detections.resize(nms_params.max_detections);
    }

    // 将坐标映射回原图空间
    for (auto& det : all_detections) {
        scale_bbox(det.bbox, letterbox_params, orig_width, orig_height);
    }

    detections = std::move(all_detections);
}

void DetectionDecoder::nms(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) {
        return;
    }

    // 按置信度降序排序
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            // 同类别才进行 NMS
            if (detections[i].class_id != detections[j].class_id) {
                continue;
            }

            float iou = compute_iou(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // 移除被抑制的检测
    std::vector<Detection> filtered;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i]) {
            filtered.push_back(std::move(detections[i]));
        }
    }

    detections = std::move(filtered);
}

float DetectionDecoder::compute_iou(const float bbox1[4], const float bbox2[4]) {
    float x1 = std::max(bbox1[0], bbox2[0]);
    float y1 = std::max(bbox1[1], bbox2[1]);
    float x2 = std::min(bbox1[2], bbox2[2]);
    float y2 = std::min(bbox1[3], bbox2[3]);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

    float union_area = area1 + area2 - intersection;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection / union_area;
}

void DetectionDecoder::scale_bbox(float bbox[4],
                                   const LetterboxParams& params,
                                   int orig_width, int orig_height) {
    // 从 letterbox 空间映射回原图空间
    LetterboxResize::map_bbox_back(bbox, params);

    // 裁剪到原图边界
    bbox[0] = std::max(0.0f, bbox[0]);
    bbox[1] = std::max(0.0f, bbox[1]);
    bbox[2] = std::min(static_cast<float>(orig_width), bbox[2]);
    bbox[3] = std::min(static_cast<float>(orig_height), bbox[3]);

    // 转换为归一化坐标
    bbox[0] /= orig_width;
    bbox[1] /= orig_height;
    bbox[2] /= orig_width;
    bbox[3] /= orig_height;
}

}  // namespace visionpipe
