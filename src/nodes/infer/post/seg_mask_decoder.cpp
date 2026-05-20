#include "seg_mask_decoder.h"

#include <algorithm>
#include <cmath>

namespace visionpipe {

void SegMaskDecoder::decode(const Tensor& det_output,
                            const Tensor& proto_output,
                            std::vector<Detection>& detections,
                            std::vector<std::vector<uint8_t>>& masks,
                            const SegMaskParams& params,
                            const LetterboxParams& letterbox_params,
                            int orig_width, int orig_height) {
    detections.clear();
    masks.clear();

    if (!det_output.valid() || det_output.shape.size() != 3) {
        return;
    }
    if (!proto_output.valid() || proto_output.shape.size() != 4) {
        return;
    }

    // YOLOv8-seg 检测输出格式：[1, 84 + num_masks, num_anchors]
    // 84 = 4 (bbox) + 80 (classes)
    const int num_classes = static_cast<int>(det_output.shape[1]) - 4 - 32;  // 32 mask coefficients
    const int num_anchors = static_cast<int>(det_output.shape[2]);

    // 原型掩码格式：[1, num_masks, mask_h, mask_w]
    const int num_masks_proto = static_cast<int>(proto_output.shape[1]);
    const int mask_h = static_cast<int>(proto_output.shape[2]);
    const int mask_w = static_cast<int>(proto_output.shape[3]);

    if (num_classes <= 0 || num_anchors <= 0 || num_masks_proto != 32) {
        return;
    }

    std::vector<float> host_proto(proto_output.numel());
    if (proto_output.memory_type() == MemoryType::CUDA_DEVICE) {
        cudaMemcpy(host_proto.data(), proto_output.data, proto_output.nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::copy_n(static_cast<const float*>(proto_output.data), proto_output.numel(), host_proto.begin());
    }

    // 解码检测框和 mask coefficients
    std::vector<std::vector<float>> bbox_list;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<std::vector<float>> mask_coeffs;

    decode_detections(det_output, params, bbox_list, class_ids, confidences, mask_coeffs);

    // 执行 NMS
    nms(bbox_list, class_ids, confidences, mask_coeffs,
        params.nms_threshold, params.max_detections);

    // 为每个检测生成 mask
    const int num_detections = static_cast<int>(bbox_list.size());
    detections.resize(num_detections);
    masks.resize(num_detections);

    for (int i = 0; i < num_detections; ++i) {
        // 构造 Detection
        Detection& det = detections[i];
        det.bbox[0] = bbox_list[i][0];
        det.bbox[1] = bbox_list[i][1];
        det.bbox[2] = bbox_list[i][2];
        det.bbox[3] = bbox_list[i][3];
        det.class_id = class_ids[i];
        det.confidence = confidences[i];
        det.track_id = -1;

        // 计算 mask
        compute_mask(host_proto.data(), mask_coeffs[i], masks[i],
                    mask_w, mask_h, params.mask_threshold);

        // 裁剪 mask 到 bbox 范围
        crop_mask_to_bbox(masks[i], det.bbox, mask_w, mask_h,
                          params.input_width, params.input_height);

        // 缩放 bbox 到原图空间
        scale_bbox(det.bbox, letterbox_params, orig_width, orig_height);

        // 缩放 mask 到原图尺寸
        resize_mask(masks[i], mask_w, mask_h, orig_width, orig_height);
    }
}

void SegMaskDecoder::decode_detections(const Tensor& det_output,
                                       const SegMaskParams& params,
                                       std::vector<std::vector<float>>& bbox_list,
                                       std::vector<int>& class_ids,
                                       std::vector<float>& confidences,
                                       std::vector<std::vector<float>>& mask_coeffs) {
    const int num_classes = static_cast<int>(det_output.shape[1]) - 4 - 32;
    const int num_anchors = static_cast<int>(det_output.shape[2]);

    // 将 GPU 数据下载到 CPU
    std::vector<float> host_data(det_output.numel());
    if (det_output.memory_type() == MemoryType::CUDA_DEVICE) {
        cudaMemcpy(host_data.data(), det_output.data, det_output.nbytes, cudaMemcpyDeviceToHost);
    } else {
        std::copy_n(static_cast<const float*>(det_output.data), det_output.numel(), host_data.begin());
    }

    bbox_list.clear();
    class_ids.clear();
    confidences.clear();
    mask_coeffs.clear();

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
        if (max_score < params.score_threshold) {
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

        // 提取 mask coefficients (32 个)
        std::vector<float> coeffs(32);
        for (int m = 0; m < 32; ++m) {
            coeffs[m] = host_data[(4 + num_classes + m) * num_anchors + i];
        }

        bbox_list.push_back({x1, y1, x2, y2});
        class_ids.push_back(max_class);
        confidences.push_back(max_score);
        mask_coeffs.push_back(std::move(coeffs));
    }
}

void SegMaskDecoder::nms(std::vector<std::vector<float>>& bbox_list,
                         std::vector<int>& class_ids,
                         std::vector<float>& confidences,
                         std::vector<std::vector<float>>& mask_coeffs,
                         float iou_threshold,
                         int max_detections) {
    if (bbox_list.empty()) {
        return;
    }

    const size_t n = bbox_list.size();

    // 按置信度降序排序
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = static_cast<int>(i);
    }
    std::sort(indices.begin(), indices.end(),
              [&confidences](int a, int b) {
                  return confidences[a] > confidences[b];
              });

    std::vector<bool> suppressed(n, false);

    auto compute_iou = [](const std::vector<float>& a, const std::vector<float>& b) {
        float x1 = std::max(a[0], b[0]);
        float y1 = std::max(a[1], b[1]);
        float x2 = std::min(a[2], b[2]);
        float y2 = std::min(a[3], b[3]);

        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = (a[2] - a[0]) * (a[3] - a[1]);
        float area_b = (b[2] - b[0]) * (b[3] - b[1]);
        float union_area = area_a + area_b - intersection;

        return union_area > 0.0f ? intersection / union_area : 0.0f;
    };

    for (size_t i = 0; i < n; ++i) {
        if (suppressed[indices[i]]) {
            continue;
        }

        for (size_t j = i + 1; j < n; ++j) {
            if (suppressed[indices[j]]) {
                continue;
            }

            // 同类别才进行 NMS
            if (class_ids[indices[i]] != class_ids[indices[j]]) {
                continue;
            }

            float iou = compute_iou(bbox_list[indices[i]], bbox_list[indices[j]]);
            if (iou > iou_threshold) {
                suppressed[indices[j]] = true;
            }
        }
    }

    // 收集未被抑制的检测结果，并限制数量
    std::vector<std::vector<float>> new_bbox_list;
    std::vector<int> new_class_ids;
    std::vector<float> new_confidences;
    std::vector<std::vector<float>> new_mask_coeffs;

    for (size_t i = 0; i < n && static_cast<int>(new_bbox_list.size()) < max_detections; ++i) {
        if (!suppressed[indices[i]]) {
            new_bbox_list.push_back(std::move(bbox_list[indices[i]]));
            new_class_ids.push_back(class_ids[indices[i]]);
            new_confidences.push_back(confidences[indices[i]]);
            new_mask_coeffs.push_back(std::move(mask_coeffs[indices[i]]));
        }
    }

    bbox_list = std::move(new_bbox_list);
    class_ids = std::move(new_class_ids);
    confidences = std::move(new_confidences);
    mask_coeffs = std::move(new_mask_coeffs);
}

void SegMaskDecoder::compute_mask(const float* proto,
                                  const std::vector<float>& coeffs,
                                  std::vector<uint8_t>& mask,
                                  int mask_width, int mask_height,
                                  float threshold) {
    // proto: [num_masks, mask_h, mask_w] = [32, mask_h, mask_w]
    // coeffs: [num_masks] = [32]
    // mask = sigmoid(proto @ coeffs)

    const int num_masks = 32;
    const int plane_size = mask_width * mask_height;

    mask.resize(plane_size);

    for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
            float sum = 0.0f;
            for (int m = 0; m < num_masks; ++m) {
                sum += proto[m * plane_size + y * mask_width + x] * coeffs[m];
            }
            // sigmoid
            float prob = 1.0f / (1.0f + std::exp(-sum));
            mask[y * mask_width + x] = prob > threshold ? 255 : 0;
        }
    }
}

void SegMaskDecoder::crop_mask_to_bbox(std::vector<uint8_t>& mask,
                                       const float bbox[4],
                                       int mask_width, int mask_height,
                                       int input_width, int input_height) {
    const float scale_x = static_cast<float>(mask_width) / static_cast<float>(input_width);
    const float scale_y = static_cast<float>(mask_height) / static_cast<float>(input_height);

    int x1 = static_cast<int>(bbox[0] * scale_x);
    int y1 = static_cast<int>(bbox[1] * scale_y);
    int x2 = static_cast<int>(bbox[2] * scale_x);
    int y2 = static_cast<int>(bbox[3] * scale_y);

    // 裁剪到 mask 边界
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(mask_width, x2);
    y2 = std::min(mask_height, y2);

    // 将 bbox 外的区域置零
    for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
            if (x < x1 || x >= x2 || y < y1 || y >= y2) {
                mask[y * mask_width + x] = 0;
            }
        }
    }
}

void SegMaskDecoder::scale_bbox(float bbox[4],
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

void SegMaskDecoder::resize_mask(std::vector<uint8_t>& mask,
                                 int src_width, int src_height,
                                 int dst_width, int dst_height) {
    if (src_width == dst_width && src_height == dst_height) {
        return;
    }

    std::vector<uint8_t> resized(dst_width * dst_height);

    const float scale_x = static_cast<float>(src_width) / dst_width;
    const float scale_y = static_cast<float>(src_height) / dst_height;

    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            int src_x = static_cast<int>(x * scale_x);
            int src_y = static_cast<int>(y * scale_y);
            src_x = std::min(src_x, src_width - 1);
            src_y = std::min(src_y, src_height - 1);
            resized[y * dst_width + x] = mask[src_y * src_width + src_x];
        }
    }

    mask = std::move(resized);
}

float SegMaskDecoder::compute_mask_bbox_iou(const std::vector<uint8_t>& mask,
                                            const float bbox[4],
                                            int mask_width, int mask_height) {
    // bbox 是归一化坐标，转换到像素坐标
    int x1 = static_cast<int>(bbox[0] * mask_width);
    int y1 = static_cast<int>(bbox[1] * mask_height);
    int x2 = static_cast<int>(bbox[2] * mask_width);
    int y2 = static_cast<int>(bbox[3] * mask_height);

    x1 = std::max(0, std::min(x1, mask_width - 1));
    y1 = std::max(0, std::min(y1, mask_height - 1));
    x2 = std::max(0, std::min(x2, mask_width));
    y2 = std::max(0, std::min(y2, mask_height));

    int intersection = 0;
    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            if (mask[y * mask_width + x] > 0) {
                ++intersection;
            }
        }
    }

    int bbox_area = (x2 - x1) * (y2 - y1);
    int mask_area = 0;
    for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
            if (mask[y * mask_width + x] > 0) {
                ++mask_area;
            }
        }
    }

    int union_count = bbox_area + mask_area - intersection;
    return union_count > 0 ? static_cast<float>(intersection) / union_count : 0.0f;
}

}  // namespace visionpipe
