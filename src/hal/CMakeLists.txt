# VisionPipe HAL (Hardware Abstraction Layer)
# 硬件抽象层：IModelEngine, IExecContext, IAllocator 接口及其 NVIDIA 实现

add_library(visionpipe_hal STATIC
    stub.cpp
    # 源文件将在后续任务中添加
    # nvidia/trt_model_engine.cpp
    # nvidia/trt_exec_context.cpp
    # nvidia/cuda_allocator.cpp
    # nvidia/nvdec_codec.cpp
)

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
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# CUDA 支持
if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_hal PRIVATE ${CUDA_INCLUDE_DIRS})
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
