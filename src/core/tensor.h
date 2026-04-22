#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "core/error.h"

namespace visionpipe {

/// @brief 内存类型
enum class MemoryType {
    CUDA_DEVICE,  ///< CUDA 设备内存
    CUDA_HOST,    ///< CUDA 主机内存（pinned）
    CPU,          ///< 普通 CPU 内存
};

/// @brief 数据类型
enum class DataType {
    FLOAT32,  ///< 32-bit 浮点
    FLOAT16,  ///< 16-bit 半精度浮点
    INT32,    ///< 32-bit 整数
    INT8,     ///< 8-bit 整数
    UINT8,    ///< 无符号 8-bit 整数
};

/// @brief 设备内存分配器接口
/// @note 线程安全：实现类必须保证 alloc/free 可并发调用
class IAllocator {
public:
    /// @brief 分配内存
    /// @param bytes 分配字节数，必须 > 0
    /// @return 内存指针，失败返回 nullptr
    /// @note 默认 256 字节对齐（适配 CUDA 合作组）
    virtual void* alloc(size_t bytes) = 0;

    /// @brief 释放内存
    /// @param ptr 内存指针，nullptr 时无操作
    virtual void free(void* ptr) = 0;

    /// @brief 返回内存类型
    virtual MemoryType type() const = 0;

    virtual ~IAllocator() = default;
};

/// @brief CPU 内存分配器
class CpuAllocator : public IAllocator {
public:
    void* alloc(size_t bytes) override {
        // 256 字节对齐
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 256, bytes) != 0) {
            return nullptr;
        }
        return ptr;
    }

    void free(void* ptr) override {
        if (ptr) {
            ::free(ptr);
        }
    }

    MemoryType type() const override { return MemoryType::CPU; }
};

/// @brief 获取数据类型大小（字节数）
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
            return 4;
        case DataType::INT8:
            return 1;
        case DataType::UINT8:
            return 1;
        default:
            return 0;
    }
}

/// @brief 张量数据结构
struct Tensor {
    std::vector<int64_t> shape;       ///< 形状，如 {1, 3, 640, 640}
    DataType dtype = DataType::FLOAT32;  ///< 数据类型
    void* data = nullptr;             ///< 数据指针
    size_t nbytes = 0;                ///< 数据字节数
    IAllocator* allocator = nullptr;  ///< 内存分配器（不拥有）

    /// @brief 默认构造函数
    Tensor() = default;

    /// @brief 构造函数
    /// @param shape_ 形状
    /// @param dtype_ 数据类型
    /// @param allocator_ 内存分配器
    Tensor(std::vector<int64_t> shape_, DataType dtype_, IAllocator* allocator_)
        : shape(std::move(shape_))
        , dtype(dtype_)
        , allocator(allocator_) {
        nbytes = compute_nbytes();
        if (allocator && nbytes > 0) {
            data = allocator->alloc(nbytes);
            if (!data) {
                throw VisionPipeError("Tensor allocation failed");
            }
        }
    }

    /// @brief 析构函数
    ~Tensor() {
        if (data && allocator) {
            allocator->free(data);
            data = nullptr;
        }
    }

    // 禁止拷贝
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 允许移动
    Tensor(Tensor&& other) noexcept
        : shape(std::move(other.shape))
        , dtype(other.dtype)
        , data(other.data)
        , nbytes(other.nbytes)
        , allocator(other.allocator) {
        other.data = nullptr;
        other.nbytes = 0;
        other.allocator = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            if (data && allocator) {
                allocator->free(data);
            }
            // 移动资源
            shape = std::move(other.shape);
            dtype = other.dtype;
            data = other.data;
            nbytes = other.nbytes;
            allocator = other.allocator;
            // 清空源对象
            other.data = nullptr;
            other.nbytes = 0;
            other.allocator = nullptr;
        }
        return *this;
    }

    /// @brief 计算元素总数
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (auto dim : shape) {
            n *= dim;
        }
        return n;
    }

    /// @brief 计算字节数
    size_t compute_nbytes() const {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }

    /// @brief 是否有效
    bool valid() const { return data != nullptr && nbytes > 0; }

    /// @brief 获取内存类型
    MemoryType memory_type() const {
        return allocator ? allocator->type() : MemoryType::CPU;
    }
};

/// @brief Tensor 智能指针类型
using TensorPtr = std::unique_ptr<Tensor>;

}  // namespace visionpipe
