#pragma once

#include <memory>
#include <string>

#include "core/node_base.h"
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
class ClassifierNode : public NodeBase {
public:
    /// @brief 构造函数
    /// @param engine TensorRT 模型引擎
    /// @param config 分类配置
    /// @param name 节点名称
    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const ClassifierConfig& config = ClassifierConfig(),
                            const std::string& name = "classifier");

    /// @brief 简化构造函数
    explicit ClassifierNode(std::shared_ptr<IModelEngine> engine,
                            const std::string& name);

    /// @brief 析构函数
    ~ClassifierNode() override;

    // 禁止拷贝
    ClassifierNode(const ClassifierNode&) = delete;
    ClassifierNode& operator=(const ClassifierNode&) = delete;

    // 允许移动
    ClassifierNode(ClassifierNode&&) noexcept = default;
    ClassifierNode& operator=(ClassifierNode&&) noexcept = default;

    /// @brief 处理帧
    void process(Frame& frame) override;

    /// @brief 启动节点
    void start() override;

    /// @brief 停止节点
    void stop(bool drain = true) override;

    /// @brief 等待停止完成
    void wait_stop() override;

    /// @brief 获取配置
    const ClassifierConfig& config() const { return config_; }

    /// @brief 获取 worker 数量
    size_t worker_count() const { return workers_; }

private:
    /// @brief worker 线程主循环
    void worker_loop(size_t worker_index);

    /// @brief 预处理：从 frame 中裁剪 crops 并打包成 batch tensor
    /// @param frame 输入帧
    /// @param batch_tensor 输出 batch tensor
    /// @param valid_crop_indices 有效 crop 的索引（用于处理超出边界的 bbox）
    void preprocess(Frame& frame, Tensor& batch_tensor,
                    std::vector<int>& valid_crop_indices);

    /// @brief 后处理：应用 softmax 并回写到 detections
    /// @param frame 输入帧（会被更新）
    /// @param output 推理输出 tensor
    /// @param valid_crop_indices 有效 crop 索引
    void postprocess(Frame& frame, const Tensor& output,
                     const std::vector<int>& valid_crop_indices);

    /// @brief 裁剪单个 crop 并预处理
    /// @param frame 输入帧
    /// @param det 检测结果
    /// @param crop_data 输出 crop 数据（CHW float32）
    /// @return 是否成功裁剪
    bool crop_and_preprocess(Frame& frame, const Detection& det,
                             std::vector<float>& crop_data);

    /// @brief 检查 worker 是否应该退出
    bool should_worker_exit() const;

    /// @brief 发射已准备好的帧（按顺序）
    void emit_ready_frames_locked();

    std::shared_ptr<IModelEngine> engine_;
    ClassifierConfig config_;
    size_t workers_;
    std::vector<std::unique_ptr<IExecContext>> contexts_;

    // 拥有的输入队列
    std::shared_ptr<BoundedQueue<Frame>> owned_input_queue_;

    // 帧重排序
    mutable std::mutex reorder_mutex_;
    std::unordered_map<int64_t, Frame> pending_outputs_;
    int64_t next_output_frame_id_ = 0;
    bool next_output_initialized_ = false;
    std::atomic<size_t> in_flight_frames_{0};
};

/// @brief ClassifierNode 智能指针类型
using ClassifierNodePtr = std::shared_ptr<ClassifierNode>;

}  // namespace visionpipe
