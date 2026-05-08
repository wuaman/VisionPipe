#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bindings.h"
#include "core/error.h"

namespace nb = nanobind;
using namespace visionpipe;

void bind_exceptions(nb::module_& m) {
    static nb::exception<VisionPipeError> visionpipe_error(m, "VisionPipeError");
    static nb::exception<ConfigError> config_error(m, "ConfigError", visionpipe_error.ptr());
    static nb::exception<NotFoundError> not_found_error(m, "NotFoundError", visionpipe_error.ptr());
    static nb::exception<CudaError> cuda_error(m, "CudaError", visionpipe_error.ptr());
    static nb::exception<ModelLoadError> model_load_error(m, "ModelLoadError", visionpipe_error.ptr());
    static nb::exception<InferError> infer_error(m, "InferError", visionpipe_error.ptr());
}
