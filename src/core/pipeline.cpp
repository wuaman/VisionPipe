#include "core/pipeline.h"

#include <algorithm>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "core/logger.h"
#include "core/source_node.h"

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
    , default_overflow_policy_(other.default_overflow_policy_)
    , queue_ref_counts_(std::move(other.queue_ref_counts_))
    , shared_queues_(std::move(other.shared_queues_)) {}

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
        queue_ref_counts_ = std::move(other.queue_ref_counts_);
        shared_queues_ = std::move(other.shared_queues_);
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

    auto a_node = get_node(a_name);
    auto b_node = get_node(b_name);

    if (b_node->input_queue()) {
        // b already has an input_queue.  Two possibilities:
        //   (1) Merge topology — the queue was set by a previous connect() and is
        //       owned by another node's output_queue.  Share it with `a`.
        //   (2) Self-owned queue — e.g. InferNode's owned_input_queue_ created
        //       in its constructor for standalone unit-test use.  No upstream
        //       owns it, so replace it with `a`'s output_queue instead.
        auto existing_queue = b_node->input_queue();

        std::shared_ptr<BoundedQueue<Frame>> shared_q;
        for (auto& sq : shared_queues_) {
            if (sq.get() == existing_queue) {
                shared_q = sq;
                break;
            }
        }

        bool found_upstream_owner = static_cast<bool>(shared_q);
        if (!shared_q) {
            for (auto& [name, node] : nodes_) {
                if (node->output_queue() && node->output_queue().get() == existing_queue) {
                    shared_q = node->output_queue();
                    found_upstream_owner = true;
                    break;
                }
            }
            if (shared_q) {
                shared_queues_.push_back(shared_q);
            }
        }

        if (found_upstream_owner && shared_q) {
            // Case (1): merge.  Share the upstream queue.
            a_node->set_output_queue(shared_q);

            auto* raw_q = shared_q.get();
            if (queue_ref_counts_.find(raw_q) == queue_ref_counts_.end()) {
                queue_ref_counts_[raw_q] = std::make_unique<QueueRefCount>();
                queue_ref_counts_[raw_q]->producer_count = 1;  // previous source
            }
            queue_ref_counts_[raw_q]->producer_count++;

            VP_LOG_INFO("Merge: '{}' → '{}' (shared queue, {} producers)",
                        a_name, b_name, queue_ref_counts_[raw_q]->producer_count);
        } else {
            // Case (2): self-owned input queue.  Override it with `a`'s output.
            if (!a_node->output_queue()) {
                a_node->create_output_queue(default_queue_capacity_, default_overflow_policy_);
            }
            b_node->set_input_queue(a_node->output_queue().get());
        }
    } else {
        // Normal connection: create a new output queue on a, wire to b
        if (!a_node->output_queue()) {
            a_node->create_output_queue(default_queue_capacity_, default_overflow_policy_);
        }
        b_node->set_input_queue(a_node->output_queue().get());
    }

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

    if (has_cycle()) {
        throw ConfigError("Pipeline DAG has cycle");
    }

    for (const auto& [name, node] : nodes_) {
        bool has_incoming = reverse_edges_.count(name) > 0 && !reverse_edges_.at(name).empty();
        bool has_outgoing = edges_.count(name) > 0 && !edges_.at(name).empty();

        if (!has_incoming && !has_outgoing) {
            if (node->is_source()) {
                VP_LOG_WARN("Node '{}' is isolated in pipeline '{}'", name, name_);
            } else {
                throw ConfigError(
                    fmt::format("Non-source node '{}' is isolated in pipeline '{}' "
                                "(no incoming edge and no outgoing edge)", name, name_));
            }
        }
    }
}

bool Pipeline::has_cycle() const {
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

    validate_dag();

    auto sources = source_nodes();
    if (sources.empty()) {
        throw ConfigError("Pipeline has no source nodes");
    }

    state_ = PipelineState::RUNNING;
    VP_LOG_INFO("Starting pipeline '{}' with {} nodes, {} source(s)",
                name_, nodes_.size(), sources.size());

    // Start all non-source nodes
    for (auto& [name, node] : nodes_) {
        if (!node->is_source()) {
            if (!node->input_queue()) {
                throw ConfigError(
                    fmt::format("Node '{}' has no input queue; "
                                "call connect() to wire it before start()", name));
            }
            node->start();
        }
    }

    // Mark source nodes as pipeline-managed
    for (auto& source : sources) {
        auto* src_node = dynamic_cast<SourceNode*>(source.get());
        if (src_node) {
            src_node->set_pipeline_managed(true);
        }
    }

    // Start source nodes in dedicated threads
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

    for (auto& [name, node] : nodes_) {
        node->stop(drain);
    }

    // Stop all source output queues (for non-merge)
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
    for (auto& t : source_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    source_threads_.clear();

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
        if (node->is_source()) {
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

    while (source->state() == NodeState::RUNNING ||
           source->state() == NodeState::DRAINING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    on_source_done(source);

    VP_LOG_INFO("Source node '{}' stopped", source->name());
}

void Pipeline::on_source_done(NodePtr source) {
    auto* raw_q = source->output_queue() ? source->output_queue().get() : nullptr;
    if (!raw_q) return;

    std::lock_guard<std::mutex> lock(source_done_mutex_);

    auto it = queue_ref_counts_.find(raw_q);
    if (it != queue_ref_counts_.end()) {
        // Merge scenario: only stop queue when all producers are done
        int done = ++(it->second->done_count);
        VP_LOG_DEBUG("Source '{}' done, queue done_count={}/{}",
                     source->name(), done, it->second->producer_count);
        if (done >= it->second->producer_count) {
            raw_q->stop();
            VP_LOG_INFO("All {} producers done for shared queue, stopping it",
                        it->second->producer_count);
        }
    } else {
        // Non-merge: stop queue directly (SourceNode already stops it in its own loop)
    }
}

}  // namespace visionpipe
