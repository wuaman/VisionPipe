#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/infer_node.h"
#include "core/frame.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

/// @brief 分类节点配置
struct ClassifierConfig {
    int input_width = 224;
    int input_height = 224;
    int max_batch_size = 32;
    size_t workers = 1;
    bool normalize_mean_std = true;
    std::vector<int> target_classes;  ///< 非空=二级分类（筛选匹配的 detections），空=整图分类
};

/// @brief 分类节点（双模式）
///
/// 模式 1（二级分类）：target_classes 非空时，筛选 frame.detections 中匹配类别的 bbox，
///   crop → batch 推理 → 结果写入 frame.classifications。无匹配则透传。
/// 模式 2（整图分类）：target_classes 为空时，直接对 frame.image 整图推理，
///   结果写入 frame.classifications（detection_index = -1），不依赖 detections。
class ClassifierNode : public InferNode {
public:
    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const ClassifierConfig& config = ClassifierConfig(),
                            const std::string& name = "classifier");

    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const std::string& name);

    ~ClassifierNode() override = default;

    ClassifierNode(const ClassifierNode&) = delete;
    ClassifierNode& operator=(const ClassifierNode&) = delete;

    ClassifierNode(ClassifierNode&&) noexcept = default;
    ClassifierNode& operator=(ClassifierNode&&) noexcept = default;

    const ClassifierConfig& config() const { return config_; }

protected:
    void process_batch(std::vector<Frame>& frames) override;

private:
    void infer_single_frame(Frame& frame);
    void infer_whole_image(Frame& frame);
    void infer_crops(Frame& frame);

    void preprocess_whole_image(Frame& frame, Tensor& input_tensor);
    void preprocess_crops(Frame& frame, Tensor& batch_tensor,
                          std::vector<int>& valid_det_indices);

    void postprocess_whole_image(Frame& frame, const Tensor& output);
    void postprocess_crops(Frame& frame, const Tensor& output,
                           const std::vector<int>& valid_det_indices);

    bool crop_and_preprocess(Frame& frame, const Detection& det,
                             std::vector<float>& crop_data);

    bool matches_target_class(int class_id) const;
    void get_image_dims(const Frame& frame, int& width, int& height) const;

    ClassifierConfig config_;
};

/// @brief ClassifierNode 智能指针类型
using ClassifierNodePtr = std::shared_ptr<ClassifierNode>;

}  // namespace visionpipe
