#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_exceptions(nb::module_& m);
void bind_enums(nb::module_& m);
void bind_frame(nb::module_& m);
void bind_nodes(nb::module_& m);
void bind_pipeline(nb::module_& m);
