# VisionPipe Tests
# 单元测试和集成测试

# 创建占位测试文件（任务 0.1 验证 CMake 构建用）
set(VISIONPIPE_TEST_SOURCES
    unit/cpp/test_cmake_build.cpp
)

# 如果测试源文件存在，创建测试目标
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/cpp/test_cmake_build.cpp")
    add_executable(visionpipe_tests ${VISIONPIPE_TEST_SOURCES})

    target_link_libraries(visionpipe_tests
        PRIVATE
            visionpipe_core
            GTest::gtest
            GTest::gtest_main
    )

    target_include_directories(visionpipe_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    # 注册测试
    include(GoogleTest)
    gtest_discover_tests(visionpipe_tests
        PROPERTIES
            LABELS "unit"
    )
else()
    # 创建空的占位测试（确保 ctest 可以运行空测试套件）
    add_custom_target(visionpipe_tests
        COMMAND echo "No tests available yet"
        COMMENT "Placeholder test target"
    )
endif()
