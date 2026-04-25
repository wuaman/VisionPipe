#pragma once

#include <memory>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "hal/imodel_engine.h"

namespace visionpipe {

struct TrtSharedState;

class TrtExecContext final : public IExecContext {
public:
  explicit TrtExecContext(std::shared_ptr<TrtSharedState> state);
  ~TrtExecContext() override;

  void infer(const Tensor &input, Tensor &output) override;

private:
  struct ContextDeleter {
    void operator()(nvinfer1::IExecutionContext *context) const noexcept;
  };

  static DataType to_core_dtype(nvinfer1::DataType data_type);
  static std::vector<int64_t> to_shape_vector(const nvinfer1::Dims &dims);
  static bool has_dynamic_dims(const nvinfer1::Dims &dims);
  void validate_input(const Tensor &input) const;
  nvinfer1::Dims resolve_output_dims() const;

  std::shared_ptr<TrtSharedState> state_;
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> context_;
  cudaStream_t stream_ = nullptr;
};

} // namespace visionpipe
