#include "core/py_node.h"

#include "core/logger.h"

namespace visionpipe {

PyNode::PyNode(ProcessFn process_fn, const std::string& name)
    : NodeBase(name)
    , process_fn_(std::move(process_fn)) {}

void PyNode::process(Frame& frame) {
    if (!process_fn_) {
        VP_LOG_WARN("PyNode '{}': no process callback set", name_);
        return;
    }
    process_fn_(frame);
}

}  // namespace visionpipe
