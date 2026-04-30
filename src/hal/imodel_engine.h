#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/tensor.h"

namespace visionpipe {

class IExecContext {
public:
    virtual ~IExecContext() = default;

    /// @brief 单输出推理
    virtual void infer(const Tensor& input, Tensor& output) = 0;

    /// @brief 多输出推理（用于分割等多输出模型）
    /// @param input 输入张量
    /// @param outputs 输出张量列表（由实现填充）
    virtual void infer_multi(const Tensor& input, std::vector<Tensor>& outputs) {
        // 默认实现：调用单输出推理
        outputs.resize(1);
        infer(input, outputs[0]);
    }
};

class IModelEngine {
public:
    virtual ~IModelEngine() = default;

    virtual std::unique_ptr<IExecContext> create_context() = 0;
    virtual size_t device_memory_bytes() const = 0;

    /// @brief 获取输出数量（默认为 1）
    virtual size_t output_count() const { return 1; }
};

class MockExecContext final : public IExecContext {
public:
    void infer(const Tensor&, Tensor&) override {}

    void infer_multi(const Tensor&, std::vector<Tensor>& outputs) override {
        outputs.clear();
    }
};

class MockModelEngine : public IModelEngine {
public:
    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<MockExecContext>();
    }

    size_t device_memory_bytes() const override { return 0; }
};

}  // namespace visionpipe
