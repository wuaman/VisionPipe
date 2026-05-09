#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/bounded_queue.h"
#include "core/node_base.h"

namespace visionpipe {

struct MjpegSinkConfig {
    int jpeg_quality = 85;    ///< JPEG quality 1-100
    size_t buffer_capacity = 2;
};

/// Sink node that JPEG-encodes incoming frames for MJPEG HTTP streaming.
///
/// Call pop_jpeg() from the Python layer to get the latest JPEG buffer
/// and serve it as a multipart HTTP stream (/mjpeg/{pipeline_id}).
class MjpegSink : public NodeBase {
public:
    explicit MjpegSink(const MjpegSinkConfig& config = MjpegSinkConfig(),
                       const std::string& name = "mjpeg_sink");
    ~MjpegSink() override = default;

    MjpegSink(const MjpegSink&) = delete;
    MjpegSink& operator=(const MjpegSink&) = delete;
    MjpegSink(MjpegSink&&) noexcept = default;
    MjpegSink& operator=(MjpegSink&&) noexcept = default;

    void process(Frame& frame) override;

    bool is_sink() const override { return true; }

    /// Drain one JPEG frame from the internal buffer.
    /// Returns nullopt on timeout or after stop().
    std::optional<std::vector<uint8_t>> pop_jpeg(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(500));

    const MjpegSinkConfig& config() const { return config_; }

private:
    MjpegSinkConfig config_;
    std::shared_ptr<BoundedQueue<std::vector<uint8_t>>> jpeg_queue_;
};

using MjpegSinkPtr = std::shared_ptr<MjpegSink>;

}  // namespace visionpipe
