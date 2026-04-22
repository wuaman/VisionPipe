#pragma once

#include <stdexcept>
#include <string>

namespace visionpipe {

/// @brief VisionPipe 异常基类
class VisionPipeError : public std::runtime_error {
public:
    explicit VisionPipeError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/// @brief 配置/参数错误
class ConfigError : public VisionPipeError {
public:
    explicit ConfigError(const std::string& msg)
        : VisionPipeError("ConfigError: " + msg) {}
};

/// @brief 资源未找到错误
class NotFoundError : public VisionPipeError {
public:
    explicit NotFoundError(const std::string& msg)
        : VisionPipeError("NotFoundError: " + msg) {}
};

/// @brief CUDA/GPU 错误
class CudaError : public VisionPipeError {
public:
    explicit CudaError(const std::string& msg)
        : VisionPipeError("CudaError: " + msg) {}
};

/// @brief 模型加载错误
class ModelLoadError : public VisionPipeError {
public:
    ModelLoadError(const std::string& path, const std::string& reason)
        : VisionPipeError("ModelLoadError: failed to load '" + path + "': " + reason) {}
};

/// @brief 推理错误
class InferError : public VisionPipeError {
public:
    explicit InferError(const std::string& msg)
        : VisionPipeError("InferError: " + msg) {}
};

}  // namespace visionpipe
