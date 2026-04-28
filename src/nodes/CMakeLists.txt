# VisionPipe Nodes Library
# 视频源节点、推理节点、输出节点等

set(VISIONPIPE_NODES_SOURCES
    source/file_source.cpp
    source/rtsp_source.cpp
)

# 添加推理节点源文件
if(VISIONPIPE_USE_CUDA AND VISIONPIPE_USE_TENSORRT)
    list(APPEND VISIONPIPE_NODES_SOURCES
        infer/detector_node.cpp
        infer/classifier_node.cpp
        infer/pre/letterbox_resize.cpp
        infer/post/detection_decoder.cpp
    )
endif()

add_library(visionpipe_nodes STATIC ${VISIONPIPE_NODES_SOURCES})

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

# TensorRT 支持
if(VISIONPIPE_USE_TENSORRT)
    target_include_directories(visionpipe_nodes PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(visionpipe_nodes PRIVATE visionpipe_hal)
    target_compile_definitions(visionpipe_nodes PRIVATE VISIONPIPE_USE_TENSORRT)
endif()

# OpenCV 支持
target_link_libraries(visionpipe_nodes PRIVATE ${OpenCV_LIBS})

# 编译选项
target_compile_options(visionpipe_nodes PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)