#pragma once

#include <atomic>

#include "core/node_base.h"

namespace visionpipe {

class SinkNode : public NodeBase {
public:
    explicit SinkNode(const std::string& name, bool enabled = true);
    ~SinkNode() override = default;

    SinkNode(const SinkNode&) = delete;
    SinkNode& operator=(const SinkNode&) = delete;

    SinkNode(SinkNode&& other) noexcept;
    SinkNode& operator=(SinkNode&& other) noexcept;

    bool is_sink() const override { return true; }

    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
    void set_enabled(bool v) { enabled_.store(v, std::memory_order_relaxed); }

    bool set_param(const std::string& name, const ParamValue& value) override;

protected:
    std::atomic<bool> enabled_;
};

}  // namespace visionpipe
