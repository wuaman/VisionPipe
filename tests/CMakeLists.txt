# VisionPipe Tests
# 单元测试和集成测试

set(VISIONPIPE_TEST_SOURCES
    unit/cpp/test_cmake_build.cpp
    unit/cpp/test_bounded_queue.cpp
)

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
