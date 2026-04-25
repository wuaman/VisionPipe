# VisionPipe HAL (Hardware Abstraction Layer)
# 硬件抽象层：IModelEngine, IExecContext, IAllocator 接口及其 NVIDIA 实现

set(VISIONPIPE_HAL_SOURCES
    stub.cpp
)

if(VISIONPIPE_USE_CUDA AND VISIONPIPE_USE_TENSORRT)
    list(APPEND VISIONPIPE_HAL_SOURCES
        nvidia/cuda_allocator.cpp
        nvidia/trt_model_engine.cpp
        nvidia/trt_exec_context.cpp
    )
endif()

add_library(visionpipe_hal STATIC ${VISIONPIPE_HAL_SOURCES})

target_include_directories(visionpipe_hal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(visionpipe_hal
    PUBLIC
        visionpipe_core
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# CUDA 支持
if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_hal PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(visionpipe_hal PRIVATE ${CUDA_LIBRARIES} cudart)
    target_compile_definitions(visionpipe_hal PRIVATE VISIONPIPE_USE_CUDA)
endif()

# TensorRT 支持
if(VISIONPIPE_USE_TENSORRT)
    target_include_directories(visionpipe_hal PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(visionpipe_hal PRIVATE ${TensorRT_LIBRARIES})
    target_compile_definitions(visionpipe_hal PRIVATE VISIONPIPE_USE_TENSORRT)
endif()

# 编译选项
target_compile_options(visionpipe_hal PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)