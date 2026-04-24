#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/bounded_queue.h"
#include "core/frame.h"

namespace visionpipe {

/// @brief 节点状态
enum class NodeState {
    INIT,       ///< 初始化状态
    RUNNING,    ///< 正在运行
    DRAINING,   ///< 正在排空队列
    STOPPED     ///< 已停止
};

/// @brief 节点参数值类型（用于 set_param）
using ParamValue = std::variant<int, float, double, std::string, std::vector<float>>;

/// @brief 节点统计信息
struct NodeStats {
    uint64_t processed_count = 0;  ///< 已处理帧数
    uint64_t error_count = 0;      ///< 错误计数
    double fps = 0.0;              ///< 处理帧率
    QueueStats input_queue_stats;  ///< 输入队列统计
};

/// @brief 节点基类
///
/// 所有节点（SourceNode、InferNode、SinkNode、PyNode）均继承此基类。
/// 节点间通过共享队列连接：上游节点的 output_queue 是下游节点的 input_queue。
class NodeBase {
public:
    /// @brief 构造函数
    /// @param name 节点名称（用于调试和日志）
    explicit NodeBase(const std::string& name);

    /// @brief 虚析构函数，停止工作线程
    virtual ~NodeBase();

    // 禁止拷贝
    NodeBase(const NodeBase&) = delete;
    NodeBase& operator=(const NodeBase&) = delete;

    // 允许移动
    NodeBase(NodeBase&& other) noexcept;
    NodeBase& operator=(NodeBase&& other) noexcept;

    /// @brief 处理帧数据（纯虚函数，由子类实现）
    /// @param frame 输入帧
    /// @note 子类实现应将处理结果写入 frame 或推送到 output_queue()
    virtual void process(Frame& frame) = 0;

    /// @brief 设置参数（用于热更新 ROI、阈值等）
    /// @param name 参数名称
    /// @param value 参数值
    /// @note 子类可重写以实现特定参数处理逻辑
    /// @return 是否成功设置
    virtual bool set_param(const std::string& name, const ParamValue& value);

    /// @brief 启动节点工作线程
    /// @note 节点从 input_queue 中取帧并调用 process()
    virtual void start();

    /// @brief 停止节点（触发 DRAINING 状态）
    /// @param drain 是否排空队列中的帧（默认 true）
    /// @note DRAINING 状态下继续处理已入队的帧，处理完后进入 STOPPED
    virtual void stop(bool drain = true);

    /// @brief 等待节点完全停止
    void wait_stop();

    /// @brief 获取节点名称
    const std::string& name() const { return name_; }

    /// @brief 获取节点状态
    NodeState state() const { return state_.load(); }

    /// @brief 获取输入队列（原始指针）
    /// @note 对于非源节点，连接后 input_queue_ 指向上游节点的 output_queue
    BoundedQueue<Frame>* input_queue() { return input_queue_; }

    /// @brief 获取输出队列（用于连接下游节点）
    /// @note 源节点和中间节点有输出队列，Sink 节点无输出队列
    std::shared_ptr<BoundedQueue<Frame>> output_queue() { return output_queue_; }

    /// @brief 设置输入队列（由 Pipeline::connect() 调用）
    void set_input_queue(BoundedQueue<Frame>* queue) { input_queue_ = queue; }

    /// @brief 创建输出队列
    /// @param capacity 队列容量
    /// @param policy 溢出策略
    void create_output_queue(size_t capacity = 16,
                             OverflowPolicy policy = OverflowPolicy::DROP_OLDEST);

    /// @brief 获取节点统计信息
    NodeStats stats() const;

    /// @brief 是否为源节点（无输入队列）
    virtual bool is_source() const { return false; }

    /// @brief 是否为输出节点（无输出队列）
    virtual bool is_sink() const { return false; }

protected:
    /// @brief 工作线程主循环
    /// @note 从 input_queue 取帧，调用 process()，推送到 output_queue
    void worker_loop();

    /// @brief 处理单帧的内部方法（含错误处理）
    /// @param frame 输入帧
    /// @return 是否成功处理
    bool process_frame(Frame& frame);

    /// @brief 初始化节点（子类可重写）
    virtual void on_init() {}

    /// @brief 停止前回调（子类可重写）
    virtual void on_stop() {}

    /// @brief 处理错误（子类可重写）
    /// @param frame 出错的帧
    /// @param error 错误信息
    virtual void on_error(Frame& frame, const std::string& error);

    // 节点名称
    std::string name_;

    // 输入队列指针（指向上游节点的 output_queue，或 nullptr 对于源节点）
    BoundedQueue<Frame>* input_queue_;

    // 输出队列（拥有所有权）
    std::shared_ptr<BoundedQueue<Frame>> output_queue_;

    // 节点状态
    std::atomic<NodeState> state_;

    // 工作线程
    std::thread worker_thread_;

    // 统计计数器
    std::atomic<uint64_t> processed_count_{0};
    std::atomic<uint64_t> error_count_{0};

    // 参数存储（用于热更新）
    std::unordered_map<std::string, ParamValue> params_;
    std::mutex params_mutex_;

    // 帧率计算
    std::atomic<int64_t> last_frame_time_{0};
    std::atomic<uint64_t> frames_since_last_fps_{0};
    std::atomic<double> current_fps_{0.0};
};

/// @brief NodeBase 智能指针类型
using NodePtr = std::shared_ptr<NodeBase>;

}  // namespace visionpipe