# VisionPipe Core Library
# 静态库：包含 Pipeline, NodeBase, BoundedQueue, Frame, Tensor 等核心组件

add_library(visionpipe_core STATIC
    stub.cpp
    logger.cpp
    # 源文件将在后续任务中添加
    # frame.cpp (header-only)
    # bounded_queue.cpp (header-only)
    # tensor.cpp (header-only)
    # node_base.cpp
    # pipeline.cpp
    # pipeline_manager.cpp
    # model_registry.cpp
)

target_include_directories(visionpipe_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(visionpipe_core
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# CUDA 支持
if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_core PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(visionpipe_core PRIVATE VISIONPIPE_USE_CUDA)
endif()

# TensorRT 支持
if(VISIONPIPE_USE_TENSORRT)
    target_include_directories(visionpipe_core PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(visionpipe_core PRIVATE ${TensorRT_LIBRARIES})
    target_compile_definitions(visionpipe_core PRIVATE VISIONPIPE_USE_TENSORRT)
endif()

# OpenCV 支持
target_link_libraries(visionpipe_core PRIVATE ${OpenCV_LIBS})

# 编译选项
target_compile_options(visionpipe_core PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)
