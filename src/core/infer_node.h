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

/// @brief 推理节点基类
///
/// 提供并行 worker 线程管理、帧重排序、drain 传播等通用推理基础设施。
/// 子类只需实现 infer_frame()，将 pre+infer+post 逻辑封装在其中。
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

protected:
    /// @brief 每帧推理入口，由 worker 线程调用
    ///
    /// 子类在此方法中执行完整的 preprocess → infer → postprocess 流程。
    /// 抛出异常时，InferNode 会增加 error_count_ 并记录日志，帧被丢弃。
    virtual void infer_frame(IExecContext& ctx, Frame& frame) = 0;

    std::shared_ptr<IModelEngine> engine_;
    size_t workers_;
    std::vector<std::unique_ptr<IExecContext>> contexts_;

private:
    void worker_loop(size_t worker_index);
    void emit_ready_frames_locked();
    bool should_worker_exit() const;

    std::shared_ptr<BoundedQueue<Frame>> owned_input_queue_;

    mutable std::mutex reorder_mutex_;
    std::unordered_map<int64_t, Frame> pending_outputs_;
    int64_t next_output_frame_id_ = 0;
    bool next_output_initialized_ = false;
    std::atomic<size_t> in_flight_frames_{0};
};

}  // namespace visionpipe
