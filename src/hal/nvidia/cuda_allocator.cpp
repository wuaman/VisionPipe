#include "hal/nvidia/cuda_allocator.h"

#include <cuda_runtime_api.h>

#include <spdlog/spdlog.h>

#include "core/error.h"

namespace visionpipe {

void *CudaAllocator::alloc(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  void *ptr = nullptr;
  const auto status = cudaMalloc(&ptr, bytes);
  if (status != cudaSuccess) {
    throw CudaError(std::string("cudaMalloc failed: ") +
                    cudaGetErrorString(status));
  }
  return ptr;
}

void CudaAllocator::free(void *ptr) {
  if (!ptr) {
    return;
  }

  const auto status = cudaFree(ptr);
  if (status != cudaSuccess) {
    spdlog::error("cudaFree failed: {}", cudaGetErrorString(status));
  }
}

} // namespace visionpipe
