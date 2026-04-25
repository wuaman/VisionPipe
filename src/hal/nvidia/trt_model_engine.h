#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInferRuntime.h>

#include "hal/imodel_engine.h"

namespace visionpipe {

class CudaAllocator;
class TrtExecContext;

struct TrtBindingInfo {
  std::string name;
  nvinfer1::DataType data_type;
  nvinfer1::Dims dims;
  bool is_input = false;
  bool is_dynamic = false;
};

struct TrtRuntimeDeleter {
  void operator()(nvinfer1::IRuntime *runtime) const noexcept;
};

struct TrtEngineDeleter {
  void operator()(nvinfer1::ICudaEngine *engine) const noexcept;
};

struct TrtSharedState {
  std::shared_ptr<CudaAllocator> allocator;
  std::unique_ptr<nvinfer1::IRuntime, TrtRuntimeDeleter> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtEngineDeleter> engine;
  TrtBindingInfo input;
  TrtBindingInfo output;
  bool has_dynamic_shapes = false;
};

class TrtModelEngine final : public IModelEngine {
public:
  explicit TrtModelEngine(const std::string &engine_path);
  ~TrtModelEngine() override = default;

  std::unique_ptr<IExecContext> create_context() override;
  size_t device_memory_bytes() const override;

private:
  struct TrtLogger final : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
  };

  static TrtBindingInfo read_binding_info(const nvinfer1::ICudaEngine &engine,
                                          const char *name);

  static TrtLogger logger_;
  std::shared_ptr<TrtSharedState> state_;

  friend class TrtExecContext;
};

} // namespace visionpipe
