#pragma once

#include "core/tensor.h"

namespace visionpipe {

class CudaAllocator final : public IAllocator {
public:
  void *alloc(size_t bytes) override;
  void free(void *ptr) override;
  MemoryType type() const override { return MemoryType::CUDA_DEVICE; }
};

} // namespace visionpipe
