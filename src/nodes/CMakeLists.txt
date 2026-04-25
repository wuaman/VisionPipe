# VisionPipe Nodes Library
# 视频源节点、推理节点、输出节点等

add_library(visionpipe_nodes STATIC
    source/file_source.cpp
    source/rtsp_source.cpp
)

target_include_directories(visionpipe_nodes
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(visionpipe_nodes
    PUBLIC
        visionpipe_core
        spdlog::spdlog
)

# CUDA 支持
if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_nodes PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(visionpipe_nodes PRIVATE VISIONPIPE_USE_CUDA)
endif()

# OpenCV 支持
target_link_libraries(visionpipe_nodes PRIVATE ${OpenCV_LIBS})

# 编译选项
target_compile_options(visionpipe_nodes PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)