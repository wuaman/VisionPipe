#pragma once

#include <atomic>
#include <chrono>
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
/// 提供并行 worker 线程管理、动态攒帧、帧重排序、drain 传播等通用推理基础设施。
/// 子类实现 process_batch()，在其中调用 run_inference() 执行推理。
class InferNode : public NodeBase {
public:
    explicit InferNode(std::shared_ptr<IModelEngine> engine,
                       size_t workers = 1,
                       size_t max_batch_size = 1,
                       std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(5),
                       const std::string& name = "infer");
    ~InferNode() override;

    void process(Frame& frame) override;
    void start() override;
    void stop(bool drain = true) override;
    void wait_stop() override;

    size_t worker_count() const { return workers_; }
    size_t max_batch_size() const { return max_batch_size_; }

protected:
    /// @brief 批量推理入口，由 worker 线程调用
    ///
    /// 子类在此方法中对批量帧执行 preprocess → infer → postprocess 流程。
    /// 可调用 run_inference() / run_inference_multi() 执行模型推理。
    virtual void process_batch(std::vector<Frame>& frames) = 0;

    /// @brief 单输出推理辅助方法（使用当前 worker 的 IExecContext）
    void run_inference(const Tensor& input, Tensor& output);

    /// @brief 多输出推理辅助方法（使用当前 worker 的 IExecContext）
    void run_inference_multi(const Tensor& input, std::vector<Tensor>& outputs);

    std::shared_ptr<IModelEngine> engine_;
    size_t workers_;
    size_t max_batch_size_;
    std::chrono::milliseconds batch_timeout_;
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

    static thread_local IExecContext* current_context_;
};

}  // namespace visionpipe
