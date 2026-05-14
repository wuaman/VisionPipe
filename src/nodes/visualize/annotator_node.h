#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/frame.h"
#include "core/node_base.h"
#include "core/tensor.h"

namespace visionpipe {

struct AnnotatorConfig {
    bool draw_detections = true;
    bool draw_tracks = true;
    bool draw_masks = true;
    float mask_alpha = 0.4f;
    std::vector<std::string> class_names;  // 空则显示 class_id
};

/// @brief 可视化标注节点
///
/// 在帧流过时原地绘制检测框、追踪 ID、分割 mask 和分类标签，
/// 后续节点直接收到带标注的 CPU 帧（HWC UINT8 BGR）。
class AnnotatorNode : public NodeBase {
public:
    explicit AnnotatorNode(const AnnotatorConfig& config = AnnotatorConfig(),
                           const std::string& name = "annotator");

    ~AnnotatorNode() override = default;

    AnnotatorNode(const AnnotatorNode&) = delete;
    AnnotatorNode& operator=(const AnnotatorNode&) = delete;
    AnnotatorNode(AnnotatorNode&&) noexcept = default;
    AnnotatorNode& operator=(AnnotatorNode&&) noexcept = default;

    void process(Frame& frame) override;

    const AnnotatorConfig& config() const { return config_; }

private:
    cv::Mat to_cpu_bgr(const Frame& frame);
    void draw_masks_overlay(cv::Mat& bgr, const Frame& frame);
    void draw_detections_overlay(cv::Mat& bgr, const Frame& frame);
    void draw_tracks_overlay(cv::Mat& bgr, const Frame& frame);
    void write_back(Frame& frame, const cv::Mat& bgr);
    std::string class_name(int id) const;
    static cv::Scalar color_for(int idx);

    AnnotatorConfig config_;
    CpuAllocator cpu_alloc_;
};

using AnnotatorNodePtr = std::shared_ptr<AnnotatorNode>;

}  // namespace visionpipe
