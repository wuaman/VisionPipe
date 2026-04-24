#include "core/pipeline.h"

#include <algorithm>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "core/logger.h"

namespace visionpipe {

Pipeline::Pipeline(const PipelineConfig& config)
    : id_(config.id.empty() ? generate_id() : config.id)
    , name_(config.name.empty() ? id_ : config.name)
    , state_(PipelineState::INIT)
    , default_queue_capacity_(config.default_queue_capacity)
    , default_overflow_policy_(config.default_overflow_policy) {}

Pipeline::~Pipeline() {
    stop(false);
    wait_stop();
}

Pipeline::Pipeline(Pipeline&& other) noexcept
    : id_(std::move(other.id_))
    , name_(std::move(other.name_))
    , state_(other.state_.load())
    , nodes_(std::move(other.nodes_))
    , edges_(std::move(other.edges_))
    , reverse_edges_(std::move(other.reverse_edges_))
    , processed_count_(other.processed_count_.load())
    , error_count_(other.error_count_.load())
    , default_queue_capacity_(other.default_queue_capacity_)
    , default_overflow_policy_(other.default_overflow_policy_) {}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        stop(false);
        wait_stop();

        id_ = std::move(other.id_);
        name_ = std::move(other.name_);
        state_ = other.state_.load();
        nodes_ = std::move(other.nodes_);
        edges_ = std::move(other.edges_);
        reverse_edges_ = std::move(other.reverse_edges_);
        processed_count_ = other.processed_count_.load();
        error_count_ = other.error_count_.load();
        default_queue_capacity_ = other.default_queue_capacity_;
        default_overflow_policy_ = other.default_overflow_policy_;
    }
    return *this;
}

std::string Pipeline::generate_id() {
    static std::atomic<uint64_t> counter{0};
    return fmt::format("pipe_{:06d}", ++counter);
}

Pipeline& Pipeline::add_node(NodePtr node) {
    if (!node) {
        throw ConfigError("Cannot add null node to pipeline");
    }

    const std::string& node_name = node->name();
    if (has_node(node_name)) {
        throw ConfigError(fmt::format("Node '{}' already exists in pipeline", node_name));
    }

    nodes_[node_name] = node;
    VP_LOG_DEBUG("Added node '{}' to pipeline '{}'", node_name, name_);

    return *this;
}

Pipeline& Pipeline::connect(NodeBase* a, NodeBase* b) {
    if (!a || !b) {
        throw ConfigError("Cannot connect null nodes");
    }

    const std::string& a_name = a->name();
    const std::string& b_name = b->name();

    if (!has_node(a_name)) {
        throw ConfigError(fmt::format("Node '{}' not found in pipeline", a_name));
    }
    if (!has_node(b_name)) {
        throw ConfigError(fmt::format("Node '{}' not found in pipeline", b_name));
    }

    // 创建上游节点的输出队列（如果还没有）
    auto a_node = get_node(a_name);
    if (!a_node->output_queue()) {
        a_node->create_output_queue(default_queue_capacity_, default_overflow_policy_);
    }

    // 将下游节点的 input_queue 指向上游节点的 output_queue
    auto b_node = get_node(b_name);
    b_node->set_input_queue(a_node->output_queue().get());

    // 记录边
    edges_[a_name].push_back(b_name);
    reverse_edges_[b_name].push_back(a_name);

    VP_LOG_DEBUG("Connected '{}' → '{}' in pipeline '{}'", a_name, b_name, name_);

    return *this;
}

Pipeline& Pipeline::connect(const NodePtr& a, const NodePtr& b) {
    return connect(a.get(), b.get());
}

void Pipeline::validate_dag() const {
    if (nodes_.empty()) {
        throw ConfigError("Pipeline has no nodes");
    }

    // 检查是否有环
    if (has_cycle()) {
        throw ConfigError("Pipeline DAG has cycle");
    }

    // 检查是否有孤立节点（无入边且无出边，且不是源节点）
    for (const auto& [name, node] : nodes_) {
        bool has_incoming = reverse_edges_.count(name) > 0 && !reverse_edges_.at(name).empty();
        bool has_outgoing = edges_.count(name) > 0 && !edges_.at(name).empty();

        if (!has_incoming && !has_outgoing) {
            VP_LOG_WARN("Node '{}' is isolated in pipeline '{}'", name, name_);
        }
    }
}

bool Pipeline::has_cycle() const {
    // Kahn's algorithm for cycle detection
    std::unordered_map<std::string, int> in_degree;
    for (const auto& [name, node] : nodes_) {
        in_degree[name] = 0;
    }

    for (const auto& [src, dsts] : edges_) {
        for (const auto& dst : dsts) {
            in_degree[dst]++;
        }
    }

    std::queue<std::string> q;
    for (const auto& [name, degree] : in_degree) {
        if (degree == 0) {
            q.push(name);
        }
    }

    size_t visited = 0;
    while (!q.empty()) {
        std::string curr = q.front();
        q.pop();
        visited++;

        if (edges_.count(curr)) {
            for (const auto& dst : edges_.at(curr)) {
                in_degree[dst]--;
                if (in_degree[dst] == 0) {
                    q.push(dst);
                }
            }
        }
    }

    return visited != nodes_.size();
}

void Pipeline::start() {
    if (state_ == PipelineState::RUNNING) {
        VP_LOG_WARN("Pipeline '{}' already running", name_);
        return;
    }

    // 验证 DAG
    validate_dag();

    // 查找源节点
    auto sources = source_nodes();
    if (sources.empty()) {
        throw ConfigError("Pipeline has no source nodes");
    }

    state_ = PipelineState::RUNNING;
    VP_LOG_INFO("Starting pipeline '{}' with {} nodes, {} source(s)",
                name_, nodes_.size(), sources.size());

    // 启动所有非源节点（它们从 input_queue 消费）
    for (auto& [name, node] : nodes_) {
        if (!node->is_source()) {
            node->start();
        }
    }

    // 启动源节点（在独立线程中运行）
    for (auto& source : sources) {
        source_threads_.emplace_back(&Pipeline::source_worker_loop, this, source);
    }
}

void Pipeline::stop(bool drain) {
    PipelineState expected = PipelineState::RUNNING;
    if (!state_.compare_exchange_strong(expected, PipelineState::DRAINING)) {
        if (state_ == PipelineState::INIT || state_ == PipelineState::STOPPED) {
            return;
        }
    }

    VP_LOG_INFO("Stopping pipeline '{}' (drain={})", name_, drain);

    // 停止所有节点
    for (auto& [name, node] : nodes_) {
        node->stop(drain);
    }

    // 停止所有源节点的输出队列
    for (auto& source : source_nodes()) {
        if (source->output_queue()) {
            source->output_queue()->stop();
        }
    }

    if (!drain) {
        state_ = PipelineState::STOPPED;
    }
}

void Pipeline::wait_stop() {
    // 等待源节点线程
    for (auto& t : source_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    source_threads_.clear();

    // 等待所有节点停止
    for (auto& [name, node] : nodes_) {
        node->wait_stop();
    }

    if (state_ == PipelineState::DRAINING) {
        state_ = PipelineState::STOPPED;
    }

    VP_LOG_INFO("Pipeline '{}' stopped, processed {} frames",
                name_, processed_count_.load());
}

bool Pipeline::has_node(const std::string& name) const {
    return nodes_.count(name) > 0;
}

NodePtr Pipeline::get_node(const std::string& name) const {
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        throw NotFoundError(fmt::format("Node '{}' not found in pipeline", name));
    }
    return it->second;
}

std::vector<NodePtr> Pipeline::source_nodes() const {
    std::vector<NodePtr> sources;
    for (const auto& [name, node] : nodes_) {
        // 源节点：无入边
        bool has_incoming = reverse_edges_.count(name) > 0 && !reverse_edges_.at(name).empty();
        if (!has_incoming) {
            sources.push_back(node);
        }
    }
    return sources;
}

PipelineStats Pipeline::stats() const {
    PipelineStats s;
    s.state = state_.load();
    s.total_frames_processed = processed_count_.load();
    s.total_errors = error_count_.load();

    for (const auto& [name, node] : nodes_) {
        s.node_stats.emplace_back(name, node->stats());
    }

    return s;
}

void Pipeline::source_worker_loop(NodePtr source) {
    VP_LOG_DEBUG("Source node '{}' worker started", source->name());

    source->start();

    // 等待源节点停止
    while (source->state() == NodeState::RUNNING ||
           source->state() == NodeState::DRAINING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    VP_LOG_INFO("Source node '{}' stopped", source->name());
}

}  // namespace visionpipe