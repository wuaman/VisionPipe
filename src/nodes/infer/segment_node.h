#pragma once

#include <memory>
#include <string>

#include "core/node_base.h"
#include "core/frame.h"
#include "hal/imodel_engine.h"
#include "nodes/infer/post/seg_mask_decoder.h"

namespace visionpipe {

/// @brief 分割节点配置
struct SegmentConfig {
    int input_width = 640;           ///< 模型输入宽度
    int input_height = 640;          ///< 模型输入高度
    float score_threshold = 0.25f;   ///< 检测置信度阈值
    float nms_threshold = 0.45f;     ///< NMS IoU 阈值
    float mask_threshold = 0.5f;     ///< 掩码二值化阈值
    int max_detections = 100;        ///< 最大检测数量
    size_t workers = 1;              ///< 并行 worker 数量
};

/// @brief YOLOv8-seg 实例分割节点
///
/// 实现实例分割的完整流程：
/// 1. Letterbox resize 预处理
/// 2. TensorRT 推理（双输出：检测 + 原型掩码）
/// 3. NMS 后处理
/// 4. Mask 解码和裁剪
///
/// 读取 frame.image，写入 frame.detections。
/// 分割掩码存储在用户可通过 detections 索引访问的位置。
class SegmentNode : public NodeBase {
public:
    /// @brief 构造函数
    /// @param engine TensorRT 模型引擎
    /// @param config 分割配置
    /// @param name 节点名称
    explicit SegmentNode(std::shared_ptr<IModelEngine> engine,
                         const SegmentConfig& config = SegmentConfig(),
                         const std::string& name = "segment");

    /// @brief 简化构造函数
    explicit SegmentNode(std::shared_ptr<IModelEngine> engine,
                         const std::string& name);

    /// @brief 析构函数
    ~SegmentNode() override;

    // 禁止拷贝
    SegmentNode(const SegmentNode&) = delete;
    SegmentNode& operator=(const SegmentNode&) = delete;

    // 禁止移动（包含 mutex 和 atomic 成员）
    SegmentNode(SegmentNode&&) = delete;
    SegmentNode& operator=(SegmentNode&&) = delete;

    /// @brief 处理帧
    void process(Frame& frame) override;

    /// @brief 启动节点
    void start() override;

    /// @brief 停止节点
    void stop(bool drain = true) override;

    /// @brief 等待停止完成
    void wait_stop() override;

    /// @brief 设置参数（支持热更新）
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const SegmentConfig& config() const { return config_; }

    /// @brief 获取 worker 数量
    size_t worker_count() const { return workers_; }

    /// @brief 获取最近一帧的分割掩码
    const std::vector<std::vector<uint8_t>>& last_masks() const { return last_masks_; }

private:
    /// @brief worker 线程主循环
    void worker_loop(size_t worker_index);

    /// @brief 预处理图像
    LetterboxParams preprocess(Frame& frame, Tensor& input_tensor);

    /// @brief 后处理推理结果
    void postprocess(Frame& frame, const Tensor& det_output,
                     const Tensor& proto_output,
                     const LetterboxParams& letterbox_params,
                     int orig_width, int orig_height);

    /// @brief 检查 worker 是否应该退出
    bool should_worker_exit() const;

    /// @brief 发射已准备好的帧（按顺序）
    void emit_ready_frames_locked();

    std::shared_ptr<IModelEngine> engine_;
    SegmentConfig config_;
    size_t workers_;
    std::vector<std::unique_ptr<IExecContext>> contexts_;

    // 拥有的输入队列
    std::shared_ptr<BoundedQueue<Frame>> owned_input_queue_;

    // 保护 start/stop/wait_stop 的并发访问
    mutable std::mutex lifecycle_mutex_;

    // 帧重排序
    mutable std::mutex reorder_mutex_;
    std::unordered_map<int64_t, Frame> pending_outputs_;
    int64_t next_output_frame_id_ = 0;
    bool next_output_initialized_ = false;
    std::atomic<size_t> in_flight_frames_{0};

    // 最近一帧的分割掩码
    std::vector<std::vector<uint8_t>> last_masks_;
    mutable std::mutex masks_mutex_;
};

/// @brief SegmentNode 智能指针类型
using SegmentNodePtr = std::shared_ptr<SegmentNode>;

}  // namespace visionpipe
