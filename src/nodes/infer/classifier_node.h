#pragma once

#include <memory>
#include <string>

#include "core/infer_node.h"
#include "core/frame.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

/// @brief 分类节点配置
struct ClassifierConfig {
    int input_width = 224;          ///< 模型输入宽度
    int input_height = 224;         ///< 模型输入高度
    int max_batch_size = 32;        ///< 最大帧内 batch 大小
    size_t workers = 1;             ///< 并行 worker 数量
    bool normalize_mean_std = true; ///< 是否使用 mean/std 归一化
};

/// @brief 分类节点
///
/// 对 DetectorNode 输出的每个检测框进行细粒度分类：
/// 1. 读取 frame.detections
/// 2. 按 bbox 从 frame.image 裁剪 crop
/// 3. 所有 crop 打包成 batch 进行推理
/// 4. 结果回写到 detections[i].class_id 和 confidence
///
/// 若 detections 为空，直接透传 frame，不做推理。
class ClassifierNode : public InferNode {
public:
    /// @brief 构造函数
    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const ClassifierConfig& config = ClassifierConfig(),
                            const std::string& name = "classifier");

    /// @brief 简化构造函数
    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const std::string& name);

    ~ClassifierNode() override = default;

    // 禁止拷贝
    ClassifierNode(const ClassifierNode&) = delete;
    ClassifierNode& operator=(const ClassifierNode&) = delete;

    // 允许移动
    ClassifierNode(ClassifierNode&&) noexcept = default;
    ClassifierNode& operator=(ClassifierNode&&) noexcept = default;

    /// @brief 获取配置
    const ClassifierConfig& config() const { return config_; }

protected:
    void infer_frame(IExecContext& ctx, Frame& frame) override;

private:
    /// @brief 预处理：从 frame 中裁剪 crops 并打包成 batch tensor
    void preprocess(Frame& frame, Tensor& batch_tensor,
                    std::vector<int>& valid_crop_indices);

    /// @brief 后处理：应用 softmax 并回写到 detections
    void postprocess(Frame& frame, const Tensor& output,
                     const std::vector<int>& valid_crop_indices);

    /// @brief 裁剪单个 crop 并预处理
    bool crop_and_preprocess(Frame& frame, const Detection& det,
                             std::vector<float>& crop_data);

    ClassifierConfig config_;
};

/// @brief ClassifierNode 智能指针类型
using ClassifierNodePtr = std::shared_ptr<ClassifierNode>;

}  // namespace visionpipe
