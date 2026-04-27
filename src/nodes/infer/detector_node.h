#pragma once

#include <memory>
#include <string>

#include "core/node_base.h"
#include "core/frame.h"
#include "nodes/infer/pre/letterbox_resize.h"
#include "nodes/infer/post/detection_decoder.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

/// @brief 检测节点配置
struct DetectorConfig {
    int input_width = 640;           ///< 模型输入宽度
    int input_height = 640;          ///< 模型输入高度
    float score_threshold = 0.25f;   ///< 置信度阈值
    float nms_threshold = 0.45f;     ///< NMS IoU 阈值
    int max_detections = 300;        ///< 最大检测数量
    size_t workers = 1;              ///< 并行 worker 数量
};

/// @brief YOLOv8 检测节点
///
/// 实现目标检测的完整流程：
/// 1. Letterbox resize 预处理
/// 2. TensorRT 推理
/// 3. NMS 后处理
/// 4. 坐标映射回原图空间
class DetectorNode : public NodeBase {
public:
    /// @brief 构造函数
    /// @param engine TensorRT 模型引擎
    /// @param config 检测配置
    /// @param name 节点名称
    explicit DetectorNode(std::shared_ptr<IModelEngine> engine,
                          const DetectorConfig& config = DetectorConfig(),
                          const std::string& name = "detector");

    /// @brief 简化构造函数
    /// @param engine TensorRT 模型引擎
    /// @param name 节点名称
    explicit DetectorNode(std::shared_ptr<IModelEngine> engine,
                          const std::string& name);

    /// @brief 析构函数
    ~DetectorNode() override;

    // 禁止拷贝
    DetectorNode(const DetectorNode&) = delete;
    DetectorNode& operator=(const DetectorNode&) = delete;

    // 允许移动
    DetectorNode(DetectorNode&&) noexcept = default;
    DetectorNode& operator=(DetectorNode&&) noexcept = default;

    /// @brief 处理帧
    /// @param frame 输入帧
    void process(Frame& frame) override;

    /// @brief 启动节点
    void start() override;

    /// @brief 停止节点
    void stop(bool drain = true) override;

    /// @brief 等待停止完成
    void wait_stop() override;

    /// @brief 设置参数（支持热更新）
    /// @param name 参数名称
    /// @param value 参数值
    /// @return 是否成功设置
    bool set_param(const std::string& name, const ParamValue& value) override;

    /// @brief 获取配置
    const DetectorConfig& config() const { return config_; }

    /// @brief 设置 ROI（感兴趣区域）
    /// @param polygons ROI 多边形顶点列表，坐标归一化到 [0, 1]
    /// @note 只输出 ROI 内的检测结果
    void set_roi(const std::vector<std::vector<float>>& polygons);

    /// @brief 清除 ROI
    void clear_roi();

    /// @brief 获取 worker 数量
    size_t worker_count() const { return workers_; }

private:
    /// @brief worker 线程主循环
    void worker_loop(size_t worker_index);

    /// @brief 预处理图像
    /// @param frame 输入帧
    /// @param input_tensor 输出预处理后的 tensor
    /// @return Letterbox 参数
    LetterboxParams preprocess(Frame& frame, Tensor& input_tensor);

    /// @brief 后处理推理结果
    /// @param frame 输入帧（会被更新）
    /// @param output 推理输出 tensor
    /// @param letterbox_params letterbox 参数
    /// @param orig_width 原图宽度
    /// @param orig_height 原图高度
    void postprocess(Frame& frame, const Tensor& output,
                     const LetterboxParams& letterbox_params,
                     int orig_width, int orig_height);

    /// @brief 检查检测结果是否在 ROI 内
    bool is_in_roi(const Detection& det) const;

    /// @brief 检查 worker 是否应该退出
    bool should_worker_exit() const;

    /// @brief 发射已准备好的帧（按顺序）
    void emit_ready_frames_locked();

    std::shared_ptr<IModelEngine> engine_;
    DetectorConfig config_;
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

    // ROI 多边形（归一化坐标）
    std::vector<std::vector<cv::Point2f>> roi_polygons_;
    std::mutex roi_mutex_;
};

/// @brief DetectorNode 智能指针类型
using DetectorNodePtr = std::shared_ptr<DetectorNode>;

}  // namespace visionpipe
