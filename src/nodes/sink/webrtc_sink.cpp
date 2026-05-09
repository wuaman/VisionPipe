#include "nodes/sink/webrtc_sink.h"

#ifdef VISIONPIPE_USE_WEBRTC

#include <rtc/rtc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

#ifdef VISIONPIPE_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/logger.h"
#include "core/tensor.h"

namespace visionpipe {

// ---------------------------------------------------------------------------
// PeerState – one WebRTC peer connection
// ---------------------------------------------------------------------------

struct PeerState {
    std::shared_ptr<rtc::PeerConnection> pc;
    std::shared_ptr<rtc::Track> track;
    std::shared_ptr<rtc::RtpPacketizationConfig> rtp_config;

    std::mutex offer_mu;
    std::condition_variable offer_cv;
    std::string offer_sdp;
    bool offer_ready = false;

    std::mutex candidates_mu;
    std::vector<std::pair<std::string, std::string>> pending_candidates;
};

// ---------------------------------------------------------------------------
// EncoderState – FFmpeg H.264 encoder
// ---------------------------------------------------------------------------

struct EncoderState {
    AVCodecContext* ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws = nullptr;
    int width = 0;
    int height = 0;

    ~EncoderState() {
        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
        if (ctx) avcodec_free_context(&ctx);
        if (sws) sws_freeContext(sws);
    }
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct WebRTCSink::Impl {
    WebRTCSinkConfig config;
    mutable std::mutex peers_mu;
    std::unordered_map<std::string, std::shared_ptr<PeerState>> peers;
    std::unique_ptr<EncoderState> encoder;
    std::mutex encoder_mu;
    std::atomic<int> frame_count{0};
    std::atomic<uint64_t> peer_seq{0};

    void ensure_encoder(int w, int h) {
        if (encoder && encoder->width == w && encoder->height == h) return;

        auto enc = std::make_unique<EncoderState>();
        enc->width = w;
        enc->height = h;

        const AVCodec* codec = nullptr;
        if (config.use_nvenc) {
            codec = avcodec_find_encoder_by_name("h264_nvenc");
        }
        if (!codec) {
            codec = avcodec_find_encoder_by_name("libx264");
        }
        if (!codec) {
            throw std::runtime_error("WebRTCSink: no H.264 encoder (h264_nvenc / libx264)");
        }

        enc->ctx = avcodec_alloc_context3(codec);
        enc->ctx->width = w;
        enc->ctx->height = h;
        enc->ctx->time_base = {1, config.fps};
        enc->ctx->framerate = {config.fps, 1};
        enc->ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        enc->ctx->bit_rate = static_cast<int64_t>(config.video_bitrate_kbps) * 1000;
        enc->ctx->gop_size = config.keyframe_interval;
        enc->ctx->max_b_frames = 0;

        // Low-latency settings
        av_opt_set(enc->ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(enc->ctx->priv_data, "tune", "zerolatency", 0);
        av_opt_set(enc->ctx->priv_data, "profile", "baseline", 0);
        // Repeat SPS/PPS in bitstream so the stream is self-contained
        av_opt_set(enc->ctx->priv_data, "x264-params", "repeat_headers=1", 0);

        if (avcodec_open2(enc->ctx, codec, nullptr) < 0) {
            throw std::runtime_error("WebRTCSink: avcodec_open2 failed");
        }

        enc->frame = av_frame_alloc();
        enc->frame->format = AV_PIX_FMT_YUV420P;
        enc->frame->width = w;
        enc->frame->height = h;
        if (av_frame_get_buffer(enc->frame, 0) < 0) {
            throw std::runtime_error("WebRTCSink: av_frame_get_buffer failed");
        }

        enc->pkt = av_packet_alloc();

        enc->sws = sws_getContext(w, h, AV_PIX_FMT_BGR24,
                                  w, h, AV_PIX_FMT_YUV420P,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!enc->sws) {
            throw std::runtime_error("WebRTCSink: sws_getContext failed");
        }

        encoder = std::move(enc);
    }

    std::vector<uint8_t> encode(const uint8_t* bgr_data, int w, int h, bool keyframe) {
        const uint8_t* src[1] = {bgr_data};
        int src_stride[1] = {w * 3};
        sws_scale(encoder->sws, src, src_stride, 0, h,
                  encoder->frame->data, encoder->frame->linesize);

        encoder->frame->pts = frame_count.load();
        if (keyframe) {
            encoder->frame->pict_type = AV_PICTURE_TYPE_I;
            encoder->frame->key_frame = 1;
        } else {
            encoder->frame->pict_type = AV_PICTURE_TYPE_NONE;
            encoder->frame->key_frame = 0;
        }

        if (avcodec_send_frame(encoder->ctx, encoder->frame) < 0) return {};

        std::vector<uint8_t> out;
        while (avcodec_receive_packet(encoder->ctx, encoder->pkt) == 0) {
            out.insert(out.end(), encoder->pkt->data,
                       encoder->pkt->data + encoder->pkt->size);
            av_packet_unref(encoder->pkt);
        }
        return out;
    }

    void broadcast(const std::vector<uint8_t>& nal, int fc) {
        // RTP 90 kHz clock: each frame advances by (90000 / fps) ticks
        const uint32_t rtp_per_frame = static_cast<uint32_t>(90000 / config.fps);
        const uint32_t ts = static_cast<uint32_t>(fc) * rtp_per_frame;

        std::vector<std::byte> bytes(nal.size());
        std::transform(nal.begin(), nal.end(), bytes.begin(),
                       [](uint8_t b) { return static_cast<std::byte>(b); });

        std::lock_guard<std::mutex> lk(peers_mu);
        for (auto& [pid, ps] : peers) {
            if (!ps->track || !ps->track->isOpen()) continue;
            try {
                ps->rtp_config->timestamp = ts;
                ps->track->send(rtc::binary(bytes.begin(), bytes.end()));
            } catch (const std::exception& e) {
                VP_LOG_WARN("WebRTCSink: send to peer {} failed: {}", pid, e.what());
            }
        }
    }
};

// ---------------------------------------------------------------------------
// WebRTCSink public API
// ---------------------------------------------------------------------------

WebRTCSink::WebRTCSink(const WebRTCSinkConfig& config, const std::string& name)
    : NodeBase(name), config_(config), impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

WebRTCSink::~WebRTCSink() = default;

std::string WebRTCSink::create_peer() {
    const uint64_t seq = ++impl_->peer_seq;
    const std::string peer_id = "peer_" + std::to_string(seq);

    auto ps = std::make_shared<PeerState>();

    rtc::Configuration rtc_cfg;
    rtc_cfg.iceServers.emplace_back(config_.stun_server);

    ps->pc = std::make_shared<rtc::PeerConnection>(rtc_cfg);

    // Build video track
    auto video_desc = rtc::Description::Video("video", rtc::Description::Direction::SendOnly);
    video_desc.addH264Codec(96);
    ps->track = ps->pc->addTrack(video_desc);

    // H.264 RTP packetizer (Annex-B input: start-code separator)
    const rtc::SSRC ssrc = static_cast<rtc::SSRC>(seq & 0xFFFF'FFFF);
    ps->rtp_config = std::make_shared<rtc::RtpPacketizationConfig>(
        ssrc, "video", 96, rtc::H264RtpPacketizer::defaultClockRate);
    auto packetizer = std::make_shared<rtc::H264RtpPacketizer>(
        rtc::H264RtpPacketizer::Separator::StartSequence, ps->rtp_config);
    ps->track->setMediaHandler(
        std::make_shared<rtc::H264PacketizationHandler>(packetizer));

    // Capture weak ref to avoid keeping peer alive in callbacks
    std::weak_ptr<PeerState> weak = ps;

    ps->pc->onLocalDescription([weak](rtc::Description desc) {
        if (auto s = weak.lock()) {
            std::lock_guard<std::mutex> lk(s->offer_mu);
            s->offer_sdp = std::string(desc);
            s->offer_ready = true;
            s->offer_cv.notify_all();
        }
    });

    ps->pc->onLocalCandidate([weak](rtc::Candidate c) {
        if (auto s = weak.lock()) {
            std::lock_guard<std::mutex> lk(s->candidates_mu);
            s->pending_candidates.emplace_back(c.candidate(), c.mid());
        }
    });

    // Trigger SDP offer generation
    ps->pc->setLocalDescription();

    std::lock_guard<std::mutex> lk(impl_->peers_mu);
    impl_->peers[peer_id] = std::move(ps);
    VP_LOG_INFO("WebRTCSink '{}': created peer {}", name(), peer_id);
    return peer_id;
}

std::string WebRTCSink::get_offer(const std::string& peer_id,
                                  std::chrono::milliseconds timeout) {
    std::shared_ptr<PeerState> ps;
    {
        std::lock_guard<std::mutex> lk(impl_->peers_mu);
        auto it = impl_->peers.find(peer_id);
        if (it == impl_->peers.end())
            throw std::runtime_error("WebRTCSink: unknown peer: " + peer_id);
        ps = it->second;
    }
    std::unique_lock<std::mutex> lk(ps->offer_mu);
    if (!ps->offer_cv.wait_for(lk, timeout, [&] { return ps->offer_ready; })) {
        throw std::runtime_error("WebRTCSink: timeout waiting for SDP offer for " + peer_id);
    }
    return ps->offer_sdp;
}

void WebRTCSink::set_answer(const std::string& peer_id, const std::string& sdp) {
    std::shared_ptr<PeerState> ps;
    {
        std::lock_guard<std::mutex> lk(impl_->peers_mu);
        auto it = impl_->peers.find(peer_id);
        if (it == impl_->peers.end())
            throw std::runtime_error("WebRTCSink: unknown peer: " + peer_id);
        ps = it->second;
    }
    ps->pc->setRemoteDescription(rtc::Description(sdp, "answer"));
}

void WebRTCSink::add_candidate(const std::string& peer_id,
                                const std::string& candidate,
                                const std::string& mid) {
    std::shared_ptr<PeerState> ps;
    {
        std::lock_guard<std::mutex> lk(impl_->peers_mu);
        auto it = impl_->peers.find(peer_id);
        if (it == impl_->peers.end()) return;
        ps = it->second;
    }
    ps->pc->addRemoteCandidate(rtc::Candidate(candidate, mid));
}

std::vector<std::pair<std::string, std::string>>
WebRTCSink::drain_candidates(const std::string& peer_id) {
    std::shared_ptr<PeerState> ps;
    {
        std::lock_guard<std::mutex> lk(impl_->peers_mu);
        auto it = impl_->peers.find(peer_id);
        if (it == impl_->peers.end()) return {};
        ps = it->second;
    }
    std::lock_guard<std::mutex> lk(ps->candidates_mu);
    return std::exchange(ps->pending_candidates, {});
}

void WebRTCSink::remove_peer(const std::string& peer_id) {
    std::lock_guard<std::mutex> lk(impl_->peers_mu);
    impl_->peers.erase(peer_id);
    VP_LOG_INFO("WebRTCSink '{}': removed peer {}", name(), peer_id);
}

int WebRTCSink::peer_count() const {
    std::lock_guard<std::mutex> lk(impl_->peers_mu);
    return static_cast<int>(impl_->peers.size());
}

void WebRTCSink::process(Frame& frame) {
    if (!frame.has_image()) return;

    {
        std::lock_guard<std::mutex> lk(impl_->peers_mu);
        if (impl_->peers.empty()) return;
    }

    const auto& img = frame.image;
    if (img.shape.size() < 2) return;

    const int h = static_cast<int>(img.shape[0]);
    const int w = static_cast<int>(img.shape[1]);

    // Bring frame to CPU as BGR
    cv::Mat bgr;

#ifdef VISIONPIPE_USE_CUDA
    if (img.memory_type() == MemoryType::CUDA_DEVICE) {
        int c = (img.shape.size() >= 3) ? static_cast<int>(img.shape[2]) : 3;
        cv::Mat cpu(h, w, (c == 3) ? CV_8UC3 : CV_8UC1);
        cudaError_t err = cudaMemcpy(cpu.data, img.data, img.nbytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            VP_LOG_ERROR("WebRTCSink '{}': cudaMemcpy failed: {}", name(), cudaGetErrorString(err));
            return;
        }
        if (c == 3) cv::cvtColor(cpu, bgr, cv::COLOR_RGB2BGR);
        else bgr = cpu;
    } else
#endif
    {
        int c = (img.shape.size() >= 3) ? static_cast<int>(img.shape[2]) : 1;
        cv::Mat src(h, w, (c == 3) ? CV_8UC3 : CV_8UC1, img.data);
        if (c == 3) cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
        else bgr = src.clone();
    }

    std::lock_guard<std::mutex> enc_lk(impl_->encoder_mu);
    try {
        impl_->ensure_encoder(w, h);
    } catch (const std::exception& e) {
        VP_LOG_ERROR("WebRTCSink '{}': encoder init failed: {}", name(), e.what());
        return;
    }

    const int fc = impl_->frame_count.load();
    const bool keyframe = (fc % config_.keyframe_interval == 0);
    auto nal = impl_->encode(bgr.data, w, h, keyframe);
    impl_->frame_count++;

    if (nal.empty()) return;
    impl_->broadcast(nal, fc);
}

}  // namespace visionpipe

// ---------------------------------------------------------------------------
// Stub implementation when VISIONPIPE_USE_WEBRTC is not defined
// ---------------------------------------------------------------------------

#else  // !VISIONPIPE_USE_WEBRTC

#include "core/logger.h"

namespace visionpipe {

struct WebRTCSink::Impl {};

WebRTCSink::WebRTCSink(const WebRTCSinkConfig& config, const std::string& name)
    : NodeBase(name), config_(config), impl_(std::make_unique<Impl>()) {
    VP_LOG_WARN("WebRTCSink '{}': built without VISIONPIPE_USE_WEBRTC; all methods are no-ops", name);
}

WebRTCSink::~WebRTCSink() = default;

void WebRTCSink::process(Frame&) {}
std::string WebRTCSink::create_peer() { return {}; }
std::string WebRTCSink::get_offer(const std::string&, std::chrono::milliseconds) { return {}; }
void WebRTCSink::set_answer(const std::string&, const std::string&) {}
void WebRTCSink::add_candidate(const std::string&, const std::string&, const std::string&) {}
std::vector<std::pair<std::string, std::string>> WebRTCSink::drain_candidates(const std::string&) { return {}; }
void WebRTCSink::remove_peer(const std::string&) {}
int WebRTCSink::peer_count() const { return 0; }

}  // namespace visionpipe

#endif  // VISIONPIPE_USE_WEBRTC
