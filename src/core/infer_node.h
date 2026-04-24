#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/bounded_queue.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "hal/imodel_engine.h"

namespace visionpipe {

class InferNode : public NodeBase {
public:
    explicit InferNode(std::shared_ptr<IModelEngine> engine,
                       size_t workers = 1,
                       const std::string& name = "infer");
    ~InferNode() override;

    void process(Frame& frame) override;
    void start() override;
    void stop(bool drain = true) override;
    void wait_stop() override;

    size_t worker_count() const { return workers_; }

private:
    void worker_loop(size_t worker_index);
    void emit_ready_frames_locked();
    bool should_worker_exit() const;

    std::shared_ptr<IModelEngine> engine_;
    size_t workers_;
    std::vector<std::unique_ptr<IExecContext>> contexts_;
    std::shared_ptr<BoundedQueue<Frame>> owned_input_queue_;

    mutable std::mutex reorder_mutex_;
    std::unordered_map<int64_t, Frame> pending_outputs_;
    int64_t next_output_frame_id_ = 0;
    bool next_output_initialized_ = false;
    std::atomic<size_t> in_flight_frames_{0};
};

}  // namespace visionpipe
