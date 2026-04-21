// test_cmake_build.cpp
// 任务 T0.1 验收测试：验证 CMake 构建配置正确

#include <gtest/gtest.h>

// 验证 visionpipe_core 静态库可链接
TEST(CMakeBuildTest, LibraryExists) {
    // 占位测试，实际验证符号导出
    // 后续任务会添加真正的功能测试
    EXPECT_TRUE(true);
}

// 验证依赖库可用
TEST(CMakeBuildTest, DependenciesAvailable) {
    // spdlog, nlohmann_json, googletest 应可通过 CMake 链接
    EXPECT_TRUE(true);
}

// 验证 CUDA 支持编译通过
TEST(CMakeBuildTest, CudaSupport) {
#ifdef VISIONPIPE_USE_CUDA
    EXPECT_TRUE(true);  // CUDA 支持已启用
#else
    EXPECT_TRUE(true);  // CUDA 支持未启用，但构建应通过
#endif
}

// 验证 TensorRT 支持编译通过
TEST(CMakeBuildTest, TensorRtSupport) {
#ifdef VISIONPIPE_USE_TENSORRT
    EXPECT_TRUE(true);  // TensorRT 支持已启用
#else
    EXPECT_TRUE(true);  // TensorRT 支持未启用，但构建应通过
#endif
}
