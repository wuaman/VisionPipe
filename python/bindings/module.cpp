#include <nanobind/nanobind.h>

#include "bindings.h"

namespace nb = nanobind;

NB_MODULE(visionpipe_python, m) {
    m.doc() = "VisionPipe Python bindings";

    bind_exceptions(m);
    bind_enums(m);
    bind_frame(m);
    bind_nodes(m);
    bind_py_node(m);
    bind_pipeline(m);
}
