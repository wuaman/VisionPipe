#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bindings.h"
#include "core/frame.h"
#include "core/node_base.h"
#include "core/py_node.h"

namespace nb = nanobind;
using namespace visionpipe;

namespace {

/// Build a ProcessFn that acquires the GIL before calling a Python callable.
/// The nb::object is captured by value; the lambda holds a strong reference.
PyNode::ProcessFn make_process_fn(nb::object callable) {
    return [callable = std::move(callable)](Frame& frame) mutable {
        nb::gil_scoped_acquire gil;
        // nb::cast gives a Python wrapper around the C++ Frame&.
        // rv_policy::reference keeps the C++ object alive (no copy/move).
        nb::object py_frame = nb::cast(frame, nb::rv_policy::reference);
        callable(py_frame);
    };
}

}  // namespace

void bind_py_node(nb::module_& m) {
    nb::class_<PyNode, NodeBase>(m, "PyNode")
        .def("__init__",
             [](PyNode* self, nb::object callable, const std::string& name) {
                 if (!PyCallable_Check(callable.ptr())) {
                     throw nb::type_error("process_fn must be callable");
                 }
                 new (self) PyNode(make_process_fn(std::move(callable)), name);
             },
             nb::arg("process_fn"),
             nb::arg("name") = "py_node",
             "Create a PyNode with a Python callable as the process function.\n"
             "The callable receives a Frame reference and may modify it in-place.")
        .def("process", [](PyNode& self, Frame& frame) { self.process(frame); },
             nb::arg("frame"));
}
