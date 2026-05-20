#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bindings.h"
#include "core/process_proxy_node.h"

namespace nb = nanobind;
using namespace visionpipe;

void bind_process_proxy_node(nb::module_& m) {
    nb::class_<ProcessProxyNode, NodeBase>(m, "ProcessProxyNode")
        .def(nb::init<const std::string&, int>(),
             nb::arg("name"), nb::arg("socket_fd"),
             "Create a ProcessProxyNode that communicates with a subprocess via UDS.")
        .def("process", &ProcessProxyNode::process, nb::arg("frame"));
}
