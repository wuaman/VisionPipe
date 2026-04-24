#pragma once

#include <cstddef>
#include <memory>

#include "core/tensor.h"

namespace visionpipe {

class IExecContext {
public:
    virtual ~IExecContext() = default;

    virtual void infer(const Tensor& input, Tensor& output) = 0;
};

class IModelEngine {
public:
    virtual ~IModelEngine() = default;

    virtual std::unique_ptr<IExecContext> create_context() = 0;
    virtual size_t device_memory_bytes() const = 0;
};

class MockExecContext final : public IExecContext {
public:
    void infer(const Tensor&, Tensor&) override {}
};

class MockModelEngine : public IModelEngine {
public:
    std::unique_ptr<IExecContext> create_context() override {
        return std::make_unique<MockExecContext>();
    }

    size_t device_memory_bytes() const override { return 0; }
};

}  // namespace visionpipe
