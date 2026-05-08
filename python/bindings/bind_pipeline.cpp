#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <unordered_map>

#include "bindings.h"
#include "core/pipeline.h"
#include "core/pipeline_builder.h"
#include "core/pipeline_manager.h"

namespace nb = nanobind;
using namespace visionpipe;

namespace {

std::unordered_map<std::string, std::shared_ptr<NodeBase>> pipeline_nodes(const Pipeline& pipeline) {
    std::unordered_map<std::string, std::shared_ptr<NodeBase>> result;
    for (const auto& [name, node] : pipeline.nodes()) {
        result.emplace(name, node);
    }
    return result;
}

}  // namespace

void bind_pipeline(nb::module_& m) {
    nb::class_<Pipeline>(m, "Pipeline")
        .def(nb::init<const PipelineConfig&>(), nb::arg("config") = PipelineConfig())
        .def("add_node", [](Pipeline& pipeline, const std::shared_ptr<NodeBase>& node) -> Pipeline& {
            return pipeline.add_node(node);
        }, nb::arg("node"), nb::rv_policy::reference_internal)
        .def("connect", [](Pipeline& pipeline,
                            const std::shared_ptr<NodeBase>& upstream,
                            const std::shared_ptr<NodeBase>& downstream) -> Pipeline& {
            return pipeline.connect(upstream, downstream);
        }, nb::arg("upstream"), nb::arg("downstream"), nb::rv_policy::reference_internal)
        .def("start", &Pipeline::start)
        .def("stop", &Pipeline::stop, nb::arg("drain") = true)
        .def("wait_stop", &Pipeline::wait_stop)
        .def("id", &Pipeline::id, nb::rv_policy::reference_internal)
        .def("name", &Pipeline::name, nb::rv_policy::reference_internal)
        .def("state", &Pipeline::state)
        .def("get_node", &Pipeline::get_node)
        .def("source_nodes", &Pipeline::source_nodes)
        .def("nodes", &pipeline_nodes)
        .def("stats", &Pipeline::stats)
        .def("processed_count", &Pipeline::processed_count)
        .def("validate_dag", &Pipeline::validate_dag);

    nb::class_<PipelineBuilder>(m, "PipelineBuilder")
        .def(nb::init<const PipelineConfig&>(), nb::arg("config") = PipelineConfig())
        .def("__rshift__", [](PipelineBuilder& builder, const std::shared_ptr<NodeBase>& node) -> PipelineBuilder& {
            return builder >> node;
        }, nb::arg("node"), nb::rv_policy::reference_internal)
        .def("build", &PipelineBuilder::build)
        .def("pipeline", &PipelineBuilder::pipeline);

    nb::class_<PipelineManager>(m, "PipelineManager")
        .def(nb::init<>())
        .def("create", nb::overload_cast<const PipelineConfig&>(&PipelineManager::create), nb::arg("config") = PipelineConfig())
        .def("create_pipeline", nb::overload_cast<PipelinePtr>(&PipelineManager::create), nb::arg("pipeline"))
        .def("start", &PipelineManager::start, nb::arg("id"))
        .def("stop", &PipelineManager::stop, nb::arg("id"), nb::arg("drain") = true)
        .def("destroy", &PipelineManager::destroy, nb::arg("id"))
        .def("status", &PipelineManager::status, nb::arg("id"))
        .def("list", &PipelineManager::list)
        .def("get", &PipelineManager::get, nb::arg("id"));
}
