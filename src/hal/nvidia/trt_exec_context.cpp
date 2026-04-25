#include "hal/nvidia/trt_exec_context.h"

#include <sstream>
#include <utility>
#include <vector>

#include "core/error.h"
#include "hal/nvidia/cuda_allocator.h"
#include "hal/nvidia/trt_model_engine.h"

namespace visionpipe {
namespace {

void throw_cuda_error(cudaError_t status, const char *action) {
  if (status != cudaSuccess) {
    throw CudaError(std::string(action) +
                    " failed: " + cudaGetErrorString(status));
  }
}

nvinfer1::Dims to_trt_dims(const std::vector<int64_t> &shape) {
  if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw InferError("input rank exceeds TensorRT maximum dimensions");
  }

  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape[static_cast<size_t>(i)];
  }
  return dims;
}

std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      os << ',';
    }
    os << shape[i];
  }
  os << ']';
  return os.str();
}

} // namespace

TrtExecContext::TrtExecContext(std::shared_ptr<TrtSharedState> state)
    : state_(std::move(state)) {
  if (!state_ || !state_->engine) {
    throw InferError("TensorRT shared state is not initialized");
  }

  context_.reset(state_->engine->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
  if (!context_) {
    throw InferError("failed to create TensorRT execution context");
  }

  throw_cuda_error(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

TrtExecContext::~TrtExecContext() {
  if (stream_ != nullptr) {
    static_cast<void>(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
}

void TrtExecContext::ContextDeleter::operator()(
    nvinfer1::IExecutionContext *context) const noexcept {
  if (context) {
    delete context;
  }
}

DataType TrtExecContext::to_core_dtype(nvinfer1::DataType data_type) {
  switch (data_type) {
  case nvinfer1::DataType::kFLOAT:
    return DataType::FLOAT32;
  case nvinfer1::DataType::kHALF:
    return DataType::FLOAT16;
  case nvinfer1::DataType::kINT32:
    return DataType::INT32;
  case nvinfer1::DataType::kINT8:
    return DataType::INT8;
  case nvinfer1::DataType::kUINT8:
    return DataType::UINT8;
  default:
    throw InferError("unsupported TensorRT tensor data type");
  }
}

std::vector<int64_t>
TrtExecContext::to_shape_vector(const nvinfer1::Dims &dims) {
  if (dims.nbDims < 0) {
    throw InferError("TensorRT returned invalid tensor rank");
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(dims.nbDims));
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    shape.push_back(dims.d[i]);
  }
  return shape;
}

bool TrtExecContext::has_dynamic_dims(const nvinfer1::Dims &dims) {
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return true;
    }
  }
  return false;
}

void TrtExecContext::validate_input(const Tensor &input) const {
  if (input.data == nullptr || input.nbytes == 0) {
    throw InferError("input tensor is empty");
  }

  if (input.memory_type() != MemoryType::CUDA_DEVICE) {
    throw InferError("TensorRT input tensor must use CUDA_DEVICE memory");
  }

  const auto expected_dtype = to_core_dtype(state_->input.data_type);
  if (input.dtype != expected_dtype) {
    throw InferError("input tensor dtype does not match TensorRT engine");
  }

  if (input.shape.size() != static_cast<size_t>(state_->input.dims.nbDims)) {
    throw InferError("input tensor rank does not match TensorRT engine");
  }

  for (size_t i = 0; i < input.shape.size(); ++i) {
    const int64_t expected_dim = state_->input.dims.d[i];
    const int64_t actual_dim = input.shape[i];
    if (actual_dim <= 0) {
      throw InferError("input tensor dimensions must be positive");
    }
    if (expected_dim >= 0 && expected_dim != actual_dim) {
      throw InferError("input tensor shape does not match TensorRT engine: "
                       "expected fixed dimension mismatch");
    }
  }

  if (input.nbytes != input.compute_nbytes()) {
    throw InferError("input tensor byte size does not match shape and dtype");
  }
}

nvinfer1::Dims TrtExecContext::resolve_output_dims() const {
  const auto dims = context_->getTensorShape(state_->output.name.c_str());
  if (dims.nbDims < 0 || has_dynamic_dims(dims)) {
    throw InferError("failed to resolve TensorRT output dimensions");
  }
  return dims;
}

void TrtExecContext::infer(const Tensor &input, Tensor &output) {
  validate_input(input);

  if (state_->input.is_dynamic) {
    const auto input_dims = to_trt_dims(input.shape);
    if (!context_->setInputShape(state_->input.name.c_str(), input_dims)) {
      throw InferError("failed to set TensorRT input shape to " +
                       shape_to_string(input.shape));
    }
  }

  if (!context_->setInputTensorAddress(state_->input.name.c_str(),
                                       input.data)) {
    throw InferError("failed to bind TensorRT input tensor address");
  }

  const int32_t shape_status = context_->inferShapes(0, nullptr);
  if (shape_status != 0) {
    throw InferError("TensorRT shape inference failed");
  }

  const auto output_dims = resolve_output_dims();
  Tensor next_output(to_shape_vector(output_dims),
                     to_core_dtype(state_->output.data_type),
                     static_cast<IAllocator *>(state_->allocator.get()));

  if (!context_->setTensorAddress(state_->output.name.c_str(),
                                  next_output.data)) {
    throw InferError("failed to bind TensorRT output tensor address");
  }

  if (!context_->enqueueV3(stream_)) {
    throw InferError("TensorRT enqueueV3 failed");
  }

  throw_cuda_error(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  output = std::move(next_output);
}

} // namespace visionpipe
