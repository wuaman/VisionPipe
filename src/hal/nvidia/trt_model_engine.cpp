#include "hal/nvidia/trt_model_engine.h"

#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/error.h"
#include "hal/nvidia/cuda_allocator.h"
#include "hal/nvidia/trt_exec_context.h"

namespace visionpipe {
namespace {

bool has_dynamic_dims(const nvinfer1::Dims &dims) {
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

std::vector<char> read_engine_blob(const std::string &engine_path) {
  std::ifstream stream(engine_path, std::ios::binary);
  if (!stream) {
    throw ModelLoadError(engine_path, "unable to open engine file");
  }

  return std::vector<char>(std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>());
}

} // namespace

TrtModelEngine::TrtLogger TrtModelEngine::logger_{};

void TrtRuntimeDeleter::operator()(nvinfer1::IRuntime *runtime) const noexcept {
  if (runtime) {
    delete runtime;
  }
}

void TrtEngineDeleter::operator()(
    nvinfer1::ICudaEngine *engine) const noexcept {
  if (engine) {
    delete engine;
  }
}

void TrtModelEngine::TrtLogger::log(Severity severity,
                                    const char *msg) noexcept {
  if (!msg) {
    return;
  }

  switch (severity) {
  case Severity::kINTERNAL_ERROR:
  case Severity::kERROR:
    spdlog::error("TensorRT: {}", msg);
    break;
  case Severity::kWARNING:
    spdlog::warn("TensorRT: {}", msg);
    break;
  case Severity::kINFO:
    spdlog::info("TensorRT: {}", msg);
    break;
  case Severity::kVERBOSE:
    spdlog::debug("TensorRT: {}", msg);
    break;
  default:
    spdlog::debug("TensorRT: {}", msg);
    break;
  }
}

TrtBindingInfo
TrtModelEngine::read_binding_info(const nvinfer1::ICudaEngine &engine,
                                  const char *name) {
  if (!name) {
    throw InferError("TensorRT engine returned null tensor name");
  }

  TrtBindingInfo binding;
  binding.name = name;
  binding.data_type = engine.getTensorDataType(name);
  binding.dims = engine.getTensorShape(name);
  binding.is_input =
      engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
  binding.is_dynamic = has_dynamic_dims(binding.dims);
  return binding;
}

TrtModelEngine::TrtModelEngine(const std::string &engine_path)
    : state_(std::make_shared<TrtSharedState>()) {
  auto engine_blob = read_engine_blob(engine_path);
  if (engine_blob.empty()) {
    throw ModelLoadError(engine_path, "engine file is empty");
  }

  state_->allocator = std::make_shared<CudaAllocator>();
  state_->runtime.reset(nvinfer1::createInferRuntime(logger_));
  if (!state_->runtime) {
    throw ModelLoadError(engine_path, "failed to create TensorRT runtime");
  }

  state_->engine.reset(state_->runtime->deserializeCudaEngine(
      engine_blob.data(), engine_blob.size()));
  if (!state_->engine) {
    throw ModelLoadError(engine_path, "failed to deserialize TensorRT engine");
  }

  int32_t input_count = 0;
  int32_t output_count = 0;
  for (int32_t i = 0; i < state_->engine->getNbIOTensors(); ++i) {
    const char *name = state_->engine->getIOTensorName(i);
    const auto binding = read_binding_info(*state_->engine, name);
    if (binding.is_input) {
      ++input_count;
      state_->input = binding;
    } else {
      ++output_count;
      state_->output = binding;
    }
  }

  if (input_count != 1 || output_count != 1) {
    throw ModelLoadError(
        engine_path,
        "only single-input single-output TensorRT engines are supported");
  }

  state_->has_dynamic_shapes =
      state_->input.is_dynamic || state_->output.is_dynamic;
}

std::unique_ptr<IExecContext> TrtModelEngine::create_context() {
  auto context = std::make_unique<TrtExecContext>(state_);
  if (!context) {
    throw InferError("failed to create TensorRT execution context");
  }
  return context;
}

size_t TrtModelEngine::device_memory_bytes() const {
  if (!state_ || !state_->engine) {
    return 0;
  }

  const int64_t bytes = state_->engine->getDeviceMemorySizeV2();
  return bytes > 0 ? static_cast<size_t>(bytes) : 0;
}

} // namespace visionpipe
