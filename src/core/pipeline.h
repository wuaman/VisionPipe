#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/error.h"
#include "core/frame.h"
#include "core/node_base.h"

namespace visionpipe {

/// @brief Pipeline 状态
enum class PipelineState {
    INIT,       ///< 初始化
    RUNNING,    ///< 运行中
    DRAINING,   ///< 排空中
    STOPPED,    ///< 已停止
    ERROR       ///< 错误状态
};

/// @brief Pipeline 配置
struct PipelineConfig {
    std::string name;                      ///< Pipeline 名称
    std::string id;                        ///< Pipeline ID（自动生成或指定）
    size_t default_queue_capacity = 16;    ///< 默认队列容量
    OverflowPolicy default_overflow_policy = OverflowPolicy::DROP_OLDEST;  ///< 默认溢出策略
};

/// @brief Pipeline 统计信息
struct PipelineStats {
    PipelineState state;
    uint64_t total_frames_processed = 0;
    uint64_t total_errors = 0;
    std::vector<std::pair<std::string, NodeStats>> node_stats;
};

/// @brief DAG 管道
///
/// 管理节点图（有向无环图），控制数据流通过节点链。
/// 节点通过有界队列连接，实现异步生产者-消费者模式。
/// 支持合并拓扑：多个 Source 共享同一个下游 input_queue。
class Pipeline {
public:
    /// @brief 构造函数
    explicit Pipeline(const PipelineConfig& config = PipelineConfig{});

    /// @brief 析构函数，停止所有节点
    ~Pipeline();

    // 禁止拷贝
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // 允许移动
    Pipeline(Pipeline&& other) noexcept;
    Pipeline& operator=(Pipeline&& other) noexcept;

    /// @brief 添加节点到 Pipeline
    Pipeline& add_node(NodePtr node);

    /// @brief 连接两个节点（a → b）
    /// 当 b 已有 input_queue 时，a 的 output_queue 指向 b 的共享 input_queue（合并拓扑）
    Pipeline& connect(NodeBase* a, NodeBase* b);

    /// @brief 连接两个节点（智能指针版本）
    Pipeline& connect(const NodePtr& a, const NodePtr& b);

    void start();
    void stop(bool drain = true);
    void wait_stop();

    const std::string& id() const { return id_; }
    const std::string& name() const { return name_; }
    PipelineState state() const { return state_.load(); }

    NodePtr get_node(const std::string& name) const;
    std::vector<NodePtr> source_nodes() const;
    const std::unordered_map<std::string, NodePtr>& nodes() const { return nodes_; }

    PipelineStats stats() const;
    uint64_t processed_count() const { return processed_count_.load(); }

    void validate_dag() const;

private:
    bool has_node(const std::string& name) const;
    bool has_cycle() const;

    void source_worker_loop(NodePtr source);

    /// @brief Source 结束后的回调：合并场景下延迟 stop 共享队列
    void on_source_done(NodePtr source);

    static std::string generate_id();

    std::string id_;
    std::string name_;
    std::atomic<PipelineState> state_;

    std::unordered_map<std::string, NodePtr> nodes_;
    std::unordered_map<std::string, std::vector<std::string>> edges_;
    std::unordered_map<std::string, std::vector<std::string>> reverse_edges_;

    std::vector<std::thread> source_threads_;

    std::atomic<uint64_t> processed_count_{0};
    std::atomic<uint64_t> error_count_{0};

    size_t default_queue_capacity_;
    OverflowPolicy default_overflow_policy_;

    // 合并拓扑：每个共享队列的生产者数和已完成数
    struct QueueRefCount {
        int producer_count = 0;
        std::atomic<int> done_count{0};
    };
    std::unordered_map<BoundedQueue<Frame>*, std::unique_ptr<QueueRefCount>> queue_ref_counts_;

    // 存储合并拓扑中共享的队列 shared_ptr，确保生命周期
    std::vector<std::shared_ptr<BoundedQueue<Frame>>> shared_queues_;

    std::mutex source_done_mutex_;
};

using PipelinePtr = std::shared_ptr<Pipeline>;

}  // namespace visionpipe