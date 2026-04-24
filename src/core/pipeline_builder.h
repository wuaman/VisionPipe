#pragma once

#include "core/node_base.h"
#include "core/pipeline.h"

namespace visionpipe {

/// @brief Pipeline 构建器
///
/// 提供流畅的 DSL 语法来构建 DAG：
/// ```cpp
/// auto src = std::make_shared<FileSource>("video.mp4");
/// auto det = std::make_shared<DetectorNode>();
/// auto sink = std::make_shared<WebRTCSink>();
///
/// PipelineBuilder builder;
/// builder >> src >> det >> sink;
/// auto pipe = builder.build();
/// ```
class PipelineBuilder {
public:
    /// @brief 构造函数
    explicit PipelineBuilder(const PipelineConfig& config = PipelineConfig{})
        : pipeline_(std::make_shared<Pipeline>(config))
        , last_node_(nullptr) {}

    /// @brief 连接节点（>> 运算符）
    /// @param node 要添加的节点
    /// @return 当前 Builder 引用（支持链式调用）
    /// @throws ConfigError 如果节点名称已存在
    PipelineBuilder& operator>>(NodePtr node) {
        if (!node) {
            throw ConfigError("Cannot add null node to builder");
        }

        // 添加节点
        pipeline_->add_node(node);

        // 如果有上一个节点，连接它们
        if (last_node_) {
            // last_node_ 是原始指针，node 是 shared_ptr
            // 需要找到 last_node_ 对应的 NodePtr
            auto last_node_ptr = pipeline_->get_node(last_node_->name());
            pipeline_->connect(last_node_ptr, node);
        }

        last_node_ = node.get();
        return *this;
    }

    /// @brief 连接现有节点（从已添加的节点继续）
    /// @param node 已在 Pipeline 中的节点
    /// @return 当前 Builder 引用
    /// @throws ConfigError 如果节点未在 Pipeline 中
    PipelineBuilder& operator>>(NodeBase* node) {
        if (!node) {
            throw ConfigError("Cannot connect null node");
        }

        if (last_node_) {
            pipeline_->connect(last_node_, node);
        }

        last_node_ = node;
        return *this;
    }

    /// @brief 从某个节点开始新的分支
    /// @param node 分支起始节点
    /// @return 新的 Builder（共享同一 Pipeline）
    PipelineBuilder branch(NodeBase* node) {
        PipelineBuilder branch_builder(pipeline_->name(), pipeline_);
        branch_builder.last_node_ = node;
        return branch_builder;
    }

    /// @brief 构建并返回 Pipeline
    /// @return Pipeline 智能指针
    /// @throws ConfigError 如果 DAG 无效
    PipelinePtr build() {
        pipeline_->validate_dag();
        return pipeline_;
    }

    /// @brief 获取当前 Pipeline（不验证）
    PipelinePtr pipeline() const { return pipeline_; }

    /// @brief 获取最后一个添加的节点
    NodeBase* last_node() const { return last_node_; }

private:
    /// @brief 私有构造函数（用于创建分支）
    PipelineBuilder(const std::string& name, PipelinePtr existing_pipeline)
        : pipeline_(existing_pipeline)
        , last_node_(nullptr) {}

    PipelinePtr pipeline_;
    NodeBase* last_node_;
};

/// @brief 全局 >> 运算符（支持 NodePtr 链式调用）
/// @param node1 第一个节点
/// @param node2 第二个节点
/// @return PipelineBuilder 对象
inline PipelineBuilder operator>>(NodePtr node1, NodePtr node2) {
    PipelineBuilder builder;
    builder >> node1 >> node2;
    return builder;
}

}  // namespace visionpipe