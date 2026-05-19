#pragma once

#include <atomic>
#include <thread>

#include "core/node_base.h"
#include "nodes/source/source_config.h"

namespace visionpipe {

class SourceNode : public NodeBase {
public:
    explicit SourceNode(const std::string& name, const SourceConfig& config);

    ~SourceNode() override;

    SourceNode(const SourceNode&) = delete;
    SourceNode& operator=(const SourceNode&) = delete;

    SourceNode(SourceNode&& other) noexcept;
    SourceNode& operator=(SourceNode&& other) noexcept;

    void start() override;
    void stop(bool drain = true) override;
    void wait_stop() override;

    bool is_source() const override { return true; }

    void process(Frame& frame) override;

    const SourceConfig& config() const { return config_; }

    void set_pipeline_managed(bool managed) { pipeline_managed_stop_ = managed; }

protected:
    virtual void on_open() = 0;
    virtual bool read_next(Frame& frame) = 0;
    virtual void on_close() {}
    virtual void on_read_error(const std::exception& e);

    void source_worker_loop();

    SourceConfig config_;

private:
    std::thread source_thread_;
    std::atomic<bool> pipeline_managed_stop_{false};
};

}  // namespace visionpipe
