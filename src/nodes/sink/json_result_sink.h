#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "core/bounded_queue.h"
#include "core/node_base.h"

namespace visionpipe {

struct JsonResultSinkConfig {
    size_t buffer_capacity = 30;
    bool include_detections = true;
    bool include_tracks = true;
};

/// Sink node that serializes per-frame detections/tracks to JSON.
///
/// Call pop_json() from the Python layer to drain JSON strings and
/// forward them to WebSocket clients.
class JsonResultSink : public NodeBase {
public:
    explicit JsonResultSink(const JsonResultSinkConfig& config = JsonResultSinkConfig(),
                            const std::string& name = "json_result_sink");
    ~JsonResultSink() override = default;

    JsonResultSink(const JsonResultSink&) = delete;
    JsonResultSink& operator=(const JsonResultSink&) = delete;
    JsonResultSink(JsonResultSink&&) noexcept = default;
    JsonResultSink& operator=(JsonResultSink&&) noexcept = default;

    void process(Frame& frame) override;

    bool is_sink() const override { return true; }

    /// Drain one JSON string from the internal buffer.
    /// Returns nullopt on timeout or after stop().
    std::optional<std::string> pop_json(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(500));

    const JsonResultSinkConfig& config() const { return config_; }

private:
    JsonResultSinkConfig config_;
    std::shared_ptr<BoundedQueue<std::string>> json_queue_;
};

using JsonResultSinkPtr = std::shared_ptr<JsonResultSink>;

}  // namespace visionpipe
