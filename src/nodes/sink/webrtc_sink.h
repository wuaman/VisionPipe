#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nodes/sink/sink_node.h"

namespace visionpipe {

struct WebRTCSinkConfig {
    int video_bitrate_kbps = 2000;
    int fps = 30;
    int keyframe_interval = 60;  ///< Keyframe every N frames
    std::string stun_server = "stun:stun.l.google.com:19302";
    bool use_nvenc = true;  ///< Prefer h264_nvenc; falls back to libx264
};

/// WebRTC sink node that streams H.264 video via libdatachannel.
///
/// The Python signaling layer drives the WebRTC handshake:
///   peer_id = sink.create_peer()
///   offer   = sink.get_offer(peer_id)        # blocks until SDP offer ready
///   # exchange with browser via WebSocket
///   sink.set_answer(peer_id, answer_sdp)
///   # poll drain_candidates() until ICE completes
///
/// Compiled-in only when VISIONPIPE_USE_WEBRTC is defined (CMake option).
/// Without it the sink compiles to no-ops so the rest of the project builds
/// without libdatachannel or FFmpeg.
class WebRTCSink : public SinkNode {
public:
    explicit WebRTCSink(const WebRTCSinkConfig& config = WebRTCSinkConfig(),
                        const std::string& name = "webrtc_sink");
    ~WebRTCSink() override;

    WebRTCSink(const WebRTCSink&) = delete;
    WebRTCSink& operator=(const WebRTCSink&) = delete;

    void process(Frame& frame) override;

    /// Allocate a new WebRTC peer connection; returns an opaque peer ID.
    std::string create_peer();

    /// Block until the local SDP offer for the peer is ready, then return it.
    std::string get_offer(const std::string& peer_id,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(10'000));

    /// Provide the browser's SDP answer.
    void set_answer(const std::string& peer_id, const std::string& sdp);

    /// Provide a remote ICE candidate received from the browser.
    void add_candidate(const std::string& peer_id,
                       const std::string& candidate,
                       const std::string& mid);

    /// Return and clear all locally generated ICE candidates for this peer.
    std::vector<std::pair<std::string, std::string>> drain_candidates(const std::string& peer_id);

    /// Tear down a peer connection.
    void remove_peer(const std::string& peer_id);

    int peer_count() const;

    const WebRTCSinkConfig& config() const { return config_; }

private:
    struct Impl;
    WebRTCSinkConfig config_;
    std::unique_ptr<Impl> impl_;
};

using WebRTCSinkPtr = std::shared_ptr<WebRTCSink>;

}  // namespace visionpipe
