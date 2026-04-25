#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "hal/nvidia/trt_model_engine.h"
#include "hal/nvidia/cuda_allocator.h"

namespace visionpipe {
namespace {

const char* get_test_engine_path() {
    return std::getenv("VISIONPIPE_TEST_TRT_ENGINE");
}

class TrtEngineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine_path_ = get_test_engine_path();
        if (!engine_path_) {
            GTEST_SKIP() << "VISIONPIPE_TEST_TRT_ENGINE not set, skipping integration tests";
        }
    }

    const char* engine_path_ = nullptr;
};

TEST_F(TrtEngineIntegrationTest, LoadEngineAndCreateContext) {
    TrtModelEngine engine(engine_path_);
    auto context = engine.create_context();
    ASSERT_NE(context, nullptr);
    EXPECT_GT(engine.device_memory_bytes(), 0u);
}

TEST_F(TrtEngineIntegrationTest, SingleInferenceProducesValidOutput) {
    TrtModelEngine engine(engine_path_);
    auto context = engine.create_context();
    ASSERT_NE(context, nullptr);

    CudaAllocator allocator;
    Tensor input({1, 3, 640, 640}, DataType::FLOAT32, &allocator);
    ASSERT_NE(input.data, nullptr);
    ASSERT_EQ(input.nbytes, 1 * 3 * 640 * 640 * 4);

    Tensor output;
    context->infer(input, output);

    EXPECT_NE(output.data, nullptr);
    EXPECT_GT(output.nbytes, 0u);
    EXPECT_FALSE(output.shape.empty());
}

TEST_F(TrtEngineIntegrationTest, TwoContextsProduceConsistentResults) {
    TrtModelEngine engine(engine_path_);
    auto ctx1 = engine.create_context();
    auto ctx2 = engine.create_context();
    ASSERT_NE(ctx1, nullptr);
    ASSERT_NE(ctx2, nullptr);

    CudaAllocator allocator;
    Tensor input({1, 3, 640, 640}, DataType::FLOAT32, &allocator);

    Tensor out1;
    Tensor out2;
    ctx1->infer(input, out1);
    ctx2->infer(input, out2);

    ASSERT_EQ(out1.shape, out2.shape);
    ASSERT_EQ(out1.dtype, out2.dtype);
    ASSERT_EQ(out1.nbytes, out2.nbytes);
}

}  // namespace
}  // namespace visionpipe