# VisionPipe Tests
# 单元测试和集成测试

set(VISIONPIPE_TEST_SOURCES
    unit/cpp/test_cmake_build.cpp
    unit/cpp/test_bounded_queue.cpp
    unit/cpp/test_logger.cpp
    unit/cpp/test_pipeline_dag.cpp
    unit/cpp/test_pipeline_manager.cpp
    unit/cpp/test_model_registry.cpp
    unit/cpp/test_parallel_workers.cpp
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

set(VISIONPIPE_INTEGRATION_TEST_SOURCES
    integration/cpp/test_trt_engine.cpp
    integration/cpp/test_source_nodes.cpp
)

# 添加推理节点测试
if(VISIONPIPE_USE_CUDA AND VISIONPIPE_USE_TENSORRT)
    list(APPEND VISIONPIPE_INTEGRATION_TEST_SOURCES
        integration/cpp/test_detector_node.cpp
    )
endif()

add_executable(visionpipe_integration_tests ${VISIONPIPE_INTEGRATION_TEST_SOURCES})

target_link_libraries(visionpipe_integration_tests
    PRIVATE
        visionpipe_hal
        visionpipe_nodes
        ${OpenCV_LIBS}
        GTest::gtest
        GTest::gtest_main
)

if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_integration_tests PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(visionpipe_integration_tests PRIVATE VISIONPIPE_USE_CUDA)
endif()

if(VISIONPIPE_USE_TENSORRT)
    target_include_directories(visionpipe_integration_tests PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(visionpipe_integration_tests PRIVATE ${TensorRT_LIBRARIES})
    target_compile_definitions(visionpipe_integration_tests PRIVATE VISIONPIPE_USE_TENSORRT)
endif()

target_include_directories(visionpipe_integration_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# 注册测试
include(GoogleTest)
gtest_discover_tests(visionpipe_tests
    PROPERTIES
        LABELS "unit"
)

gtest_discover_tests(visionpipe_integration_tests
    PROPERTIES
        LABELS "integration"
)