#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "core/bounded_queue.h"
#include "nodes/sink/sink_node.h"

namespace visionpipe {

struct JsonResultSinkConfig {
    size_t buffer_capacity = 30;
    bool include_detections = true;
    bool include_tracks = true;
};

class JsonResultSink : public SinkNode {
public:
    explicit JsonResultSink(const JsonResultSinkConfig& config = JsonResultSinkConfig(),
                            const std::string& name = "json_result_sink");
    ~JsonResultSink() override = default;

    JsonResultSink(const JsonResultSink&) = delete;
    JsonResultSink& operator=(const JsonResultSink&) = delete;
    JsonResultSink(JsonResultSink&&) noexcept = default;
    JsonResultSink& operator=(JsonResultSink&&) noexcept = default;

    void process(Frame& frame) override;

    std::optional<std::string> pop_json(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(500));

    const JsonResultSinkConfig& config() const { return config_; }

private:
    JsonResultSinkConfig config_;
    std::shared_ptr<BoundedQueue<std::string>> json_queue_;
};

using JsonResultSinkPtr = std::shared_ptr<JsonResultSink>;

}  // namespace visionpipe
