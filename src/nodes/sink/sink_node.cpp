#include "nodes/sink/sink_node.h"

#include <string>

namespace visionpipe {

SinkNode::SinkNode(const std::string& name, bool enabled)
    : NodeBase(name), enabled_(enabled) {}

SinkNode::SinkNode(SinkNode&& other) noexcept
    : NodeBase(std::move(other)), enabled_(other.enabled_.load()) {}

SinkNode& SinkNode::operator=(SinkNode&& other) noexcept {
    if (this != &other) {
        NodeBase::operator=(std::move(other));
        enabled_ = other.enabled_.load();
    }
    return *this;
}

bool SinkNode::set_param(const std::string& name, const ParamValue& value) {
    if (name == "enabled") {
        if (auto* v = std::get_if<int>(&value)) {
            set_enabled(*v != 0);
            return true;
        }
        return false;
    }
    return NodeBase::set_param(name, value);
}

}  // namespace visionpipe
