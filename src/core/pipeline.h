#pragma once

#include <atomic>
#include <memory>
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
    /// @param node 节点指针
    /// @return 当前 Pipeline 引用（支持链式调用）
    /// @throws ConfigError 如果节点名称已存在
    Pipeline& add_node(NodePtr node);

    /// @brief 连接两个节点（a → b）
    /// @param a 上游节点
    /// @param b 下游节点
    /// @return 当前 Pipeline 引用（支持链式调用）
    /// @throws ConfigError 如果节点未添加到 Pipeline 或形成环
    Pipeline& connect(NodeBase* a, NodeBase* b);

    /// @brief 连接两个节点（智能指针版本）
    Pipeline& connect(const NodePtr& a, const NodePtr& b);

    /// @brief 启动 Pipeline
    /// @throws ConfigError 如果 Pipeline 没有源节点或配置非法
    void start();

    /// @brief 停止 Pipeline（触发 DRAINING）
    /// @param drain 是否排空队列中的帧（默认 true）
    void stop(bool drain = true);

    /// @brief 等待 Pipeline 完全停止
    void wait_stop();

    /// @brief 获取 Pipeline ID
    const std::string& id() const { return id_; }

    /// @brief 获取 Pipeline 名称
    const std::string& name() const { return name_; }

    /// @brief 获取 Pipeline 状态
    PipelineState state() const { return state_.load(); }

    /// @brief 获取节点（按名称）
    /// @throws NotFoundError 如果节点不存在
    NodePtr get_node(const std::string& name) const;

    /// @brief 获取所有源节点
    std::vector<NodePtr> source_nodes() const;

    /// @brief 获取所有节点
    const std::unordered_map<std::string, NodePtr>& nodes() const { return nodes_; }

    /// @brief 获取 Pipeline 统计信息
    PipelineStats stats() const;

    /// @brief 获取已处理帧数
    uint64_t processed_count() const { return processed_count_.load(); }

    /// @brief 验证 DAG（检查是否有环、是否有孤立节点）
    /// @throws ConfigError 如果 DAG 无效
    void validate_dag() const;

private:
    /// @brief 检查节点是否存在
    bool has_node(const std::string& name) const;

    /// @brief 检查是否有环（拓扑排序检测）
    bool has_cycle() const;

    /// @brief 源节点工作线程函数
    void source_worker_loop(NodePtr source);

    /// @brief 生成唯一 ID
    static std::string generate_id();

    std::string id_;
    std::string name_;
    std::atomic<PipelineState> state_;

    // 节点映射（名称 → 节点）
    std::unordered_map<std::string, NodePtr> nodes_;

    // 边（上游 → 下游列表）
    std::unordered_map<std::string, std::vector<std::string>> edges_;

    // 反向边（下游 → 上游列表）
    std::unordered_map<std::string, std::vector<std::string>> reverse_edges_;

    // 源节点工作线程
    std::vector<std::thread> source_threads_;

    // 统计计数器
    std::atomic<uint64_t> processed_count_{0};
    std::atomic<uint64_t> error_count_{0};

    // 默认配置
    size_t default_queue_capacity_;
    OverflowPolicy default_overflow_policy_;
};

/// @brief Pipeline 智能指针类型
using PipelinePtr = std::shared_ptr<Pipeline>;

}  // namespace visionpipe