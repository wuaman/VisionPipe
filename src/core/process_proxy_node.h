#pragma once

#include <string>

#include "core/frame.h"
#include "core/node_base.h"

#include <nlohmann/json.hpp>

namespace visionpipe {

class ProcessProxyNode : public NodeBase {
public:
    explicit ProcessProxyNode(const std::string& name, int socket_fd);
    ~ProcessProxyNode() override;

    ProcessProxyNode(const ProcessProxyNode&) = delete;
    ProcessProxyNode& operator=(const ProcessProxyNode&) = delete;

    void process(Frame& frame) override;

protected:
    void on_stop() override;

private:
    int socket_fd_;

    nlohmann::json frame_to_json(const Frame& frame) const;
    void apply_updates(const nlohmann::json& response, Frame& frame);

    bool send_message(const nlohmann::json& msg);
    nlohmann::json recv_message();

    bool send_bytes(const void* data, size_t len);
    bool recv_bytes(void* data, size_t len);
};

}  // namespace visionpipe
