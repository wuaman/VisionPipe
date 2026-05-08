#include <nanobind/nanobind.h>

#include "bindings.h"
#include "core/bounded_queue.h"
#include "core/node_base.h"
#include "core/pipeline.h"
#include "core/pipeline_manager.h"
#include "nodes/source/source_config.h"

namespace nb = nanobind;
using namespace visionpipe;

void bind_enums(nb::module_& m) {
    nb::enum_<PipelineState>(m, "PipelineState")
        .value("INIT", PipelineState::INIT)
        .value("RUNNING", PipelineState::RUNNING)
        .value("DRAINING", PipelineState::DRAINING)
        .value("STOPPED", PipelineState::STOPPED)
        .value("ERROR", PipelineState::ERROR);

    nb::enum_<PipelineStatus>(m, "PipelineStatus")
        .value("INIT", PipelineStatus::INIT)
        .value("RUNNING", PipelineStatus::RUNNING)
        .value("DRAINING", PipelineStatus::DRAINING)
        .value("STOPPED", PipelineStatus::STOPPED)
        .value("ERROR", PipelineStatus::ERROR);

    nb::enum_<NodeState>(m, "NodeState")
        .value("INIT", NodeState::INIT)
        .value("RUNNING", NodeState::RUNNING)
        .value("DRAINING", NodeState::DRAINING)
        .value("STOPPED", NodeState::STOPPED);

    nb::enum_<OverflowPolicy>(m, "OverflowPolicy")
        .value("DROP_OLDEST", OverflowPolicy::DROP_OLDEST)
        .value("DROP_NEWEST", OverflowPolicy::DROP_NEWEST)
        .value("BLOCK", OverflowPolicy::BLOCK);

    nb::enum_<DecodeMode>(m, "DecodeMode")
        .value("AUTO", DecodeMode::AUTO)
        .value("GPU", DecodeMode::GPU)
        .value("CPU", DecodeMode::CPU);
}
