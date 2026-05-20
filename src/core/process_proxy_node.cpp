#include "core/process_proxy_node.h"

#include "core/error.h"
#include "core/logger.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace visionpipe {

ProcessProxyNode::ProcessProxyNode(const std::string& name, int socket_fd)
    : NodeBase(name), socket_fd_(socket_fd) {
    if (socket_fd_ < 0) {
        throw ConfigError("ProcessProxyNode requires a valid socket fd");
    }
}

ProcessProxyNode::~ProcessProxyNode() = default;

void ProcessProxyNode::process(Frame& frame) {
    auto msg = frame_to_json(frame);
    msg["type"] = "frame";

    if (!send_message(msg)) {
        VP_LOG_ERROR("ProcessProxyNode '{}': send failed", name_);
        return;
    }

    auto response = recv_message();
    if (response.is_null()) {
        VP_LOG_ERROR("ProcessProxyNode '{}': recv failed", name_);
        return;
    }

    if (response.contains("error") && !response["error"].is_null()) {
        VP_LOG_ERROR("ProcessProxyNode '{}': subprocess error: {}",
                     name_, response["error"].get<std::string>());
        return;
    }

    apply_updates(response, frame);
}

void ProcessProxyNode::on_stop() {
    nlohmann::json shutdown = {{"type", "shutdown"}};
    send_message(shutdown);
}

nlohmann::json ProcessProxyNode::frame_to_json(const Frame& frame) const {
    nlohmann::json j;
    j["stream_id"] = frame.stream_id;
    j["frame_id"] = frame.frame_id;
    j["pts_us"] = frame.pts_us;

    auto& dets = j["detections"] = nlohmann::json::array();
    for (const auto& d : frame.detections) {
        dets.push_back({
            {"bbox", {d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]}},
            {"class_id", d.class_id},
            {"confidence", d.confidence},
            {"track_id", d.track_id}
        });
    }

    auto& cls = j["classifications"] = nlohmann::json::array();
    for (const auto& c : frame.classifications) {
        cls.push_back({
            {"detection_index", c.detection_index},
            {"class_id", c.class_id},
            {"confidence", c.confidence}
        });
    }

    auto& trks = j["tracks"] = nlohmann::json::array();
    for (const auto& t : frame.tracks) {
        trks.push_back({
            {"track_id", t.track_id},
            {"class_id", t.class_id},
            {"bbox", {t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]}},
            {"age", t.age},
            {"confidence", t.confidence}
        });
    }

    auto& ud = j["user_data"] = nlohmann::json::object();
    for (const auto& [key, val] : frame.user_data) {
        if (auto* v = std::any_cast<bool>(&val))
            ud[key] = *v;
        else if (auto* v = std::any_cast<int>(&val))
            ud[key] = *v;
        else if (auto* v = std::any_cast<int64_t>(&val))
            ud[key] = *v;
        else if (auto* v = std::any_cast<float>(&val))
            ud[key] = *v;
        else if (auto* v = std::any_cast<double>(&val))
            ud[key] = *v;
        else if (auto* v = std::any_cast<std::string>(&val))
            ud[key] = *v;
    }

    return j;
}

void ProcessProxyNode::apply_updates(const nlohmann::json& response,
                                     Frame& frame) {
    if (!response.contains("user_data")) return;
    const auto& ud = response["user_data"];
    if (!ud.is_object()) return;

    for (auto it = ud.begin(); it != ud.end(); ++it) {
        const auto& key = it.key();
        const auto& val = it.value();

        if (val.is_null())
            frame.user_data.erase(key);
        else if (val.is_boolean())
            frame.user_data[key] = val.get<bool>();
        else if (val.is_number_integer())
            frame.user_data[key] = val.get<int64_t>();
        else if (val.is_number_float())
            frame.user_data[key] = val.get<double>();
        else if (val.is_string())
            frame.user_data[key] = val.get<std::string>();
        else
            frame.user_data[key] = val.dump();
    }
}

bool ProcessProxyNode::send_message(const nlohmann::json& msg) {
    std::string data = msg.dump();
    uint32_t len = static_cast<uint32_t>(data.size());
    uint8_t header[4] = {
        static_cast<uint8_t>((len >> 24) & 0xFF),
        static_cast<uint8_t>((len >> 16) & 0xFF),
        static_cast<uint8_t>((len >> 8) & 0xFF),
        static_cast<uint8_t>(len & 0xFF)
    };

    if (!send_bytes(header, 4)) return false;
    return send_bytes(data.data(), data.size());
}

nlohmann::json ProcessProxyNode::recv_message() {
    uint8_t header[4];
    if (!recv_bytes(header, 4)) return nullptr;

    uint32_t len = (static_cast<uint32_t>(header[0]) << 24) |
                   (static_cast<uint32_t>(header[1]) << 16) |
                   (static_cast<uint32_t>(header[2]) << 8) |
                   static_cast<uint32_t>(header[3]);

    if (len > 64 * 1024 * 1024) {
        VP_LOG_ERROR("ProcessProxyNode '{}': message too large ({} bytes)",
                     name_, len);
        return nullptr;
    }

    std::string data(len, '\0');
    if (!recv_bytes(data.data(), len)) return nullptr;

    try {
        return nlohmann::json::parse(data);
    } catch (const nlohmann::json::exception& e) {
        VP_LOG_ERROR("ProcessProxyNode '{}': JSON parse error: {}",
                     name_, e.what());
        return nullptr;
    }
}

bool ProcessProxyNode::send_bytes(const void* data, size_t len) {
    auto ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(socket_fd_, ptr + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool ProcessProxyNode::recv_bytes(void* data, size_t len) {
    auto ptr = static_cast<char*>(data);
    size_t received = 0;
    while (received < len) {
        ssize_t n = ::recv(socket_fd_, ptr + received, len - received, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        received += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace visionpipe
