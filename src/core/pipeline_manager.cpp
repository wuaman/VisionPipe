#include "core/pipeline_manager.h"

#include <algorithm>
#include <utility>

#include "core/error.h"
#include "core/logger.h"

namespace visionpipe {

namespace {

PipelinePtr require_pipeline(const std::unordered_map<std::string, PipelinePtr>& pipelines,
                             const std::string& id) {
    auto it = pipelines.find(id);
    if (it == pipelines.end()) {
        throw NotFoundError("Pipeline '" + id + "' not found");
    }
    return it->second;
}

}  // namespace

std::string PipelineManager::create(const PipelineConfig& config) {
    auto pipeline = std::make_shared<Pipeline>(config);
    return create(std::move(pipeline));
}

std::string PipelineManager::create(PipelinePtr pipeline) {
    if (!pipeline) {
        throw ConfigError("Cannot create null pipeline");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const std::string id = pipeline->id();
    if (pipelines_.count(id) > 0) {
        throw ConfigError("Pipeline '" + id + "' already exists");
    }

    pipelines_.emplace(id, std::move(pipeline));
    VP_LOG_INFO("Created pipeline '{}'", id);
    return id;
}

void PipelineManager::start(const std::string& id) {
    auto pipeline = get(id);
    pipeline->start();
}

void PipelineManager::stop(const std::string& id, bool drain) {
    auto pipeline = get(id);
    pipeline->stop(drain);
    if (drain) {
        pipeline->wait_stop();
    }
}

void PipelineManager::destroy(const std::string& id) {
    PipelinePtr pipeline;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pipelines_.find(id);
        if (it == pipelines_.end()) {
            throw NotFoundError("Pipeline '" + id + "' not found");
        }
        pipeline = it->second;
        pipelines_.erase(it);
    }

    pipeline->stop(false);
    pipeline->wait_stop();
    VP_LOG_INFO("Destroyed pipeline '{}'", id);
}

PipelineStatus PipelineManager::status(const std::string& id) const {
    return to_status(get(id)->state());
}

std::vector<std::string> PipelineManager::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(pipelines_.size());
    for (const auto& [id, pipeline] : pipelines_) {
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

PipelinePtr PipelineManager::get(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return require_pipeline(pipelines_, id);
}

PipelineStatus PipelineManager::to_status(PipelineState state) {
    switch (state) {
        case PipelineState::INIT:
            return PipelineStatus::INIT;
        case PipelineState::RUNNING:
            return PipelineStatus::RUNNING;
        case PipelineState::DRAINING:
            return PipelineStatus::DRAINING;
        case PipelineState::STOPPED:
            return PipelineStatus::STOPPED;
        case PipelineState::ERROR:
            return PipelineStatus::ERROR;
    }

    return PipelineStatus::ERROR;
}

}  // namespace visionpipe
