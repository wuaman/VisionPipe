#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/pipeline.h"

namespace visionpipe {

enum class PipelineStatus {
    INIT,
    RUNNING,
    DRAINING,
    STOPPED,
    ERROR
};

class PipelineManager {
public:
    PipelineManager() = default;

    std::string create(const PipelineConfig& config = PipelineConfig{});
    std::string create(PipelinePtr pipeline);
    void start(const std::string& id);
    void stop(const std::string& id, bool drain = true);
    void destroy(const std::string& id);

    PipelineStatus status(const std::string& id) const;
    std::vector<std::string> list() const;
    PipelinePtr get(const std::string& id) const;

private:
    static PipelineStatus to_status(PipelineState state);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, PipelinePtr> pipelines_;
};

}  // namespace visionpipe
