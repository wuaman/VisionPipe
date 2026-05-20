# VisionPipe Python Bindings

set(VISIONPIPE_BINDING_SOURCES
    bindings/module.cpp
    bindings/bind_exceptions.cpp
    bindings/bind_enums.cpp
    bindings/bind_frame.cpp
    bindings/bind_nodes.cpp
    bindings/bind_py_node.cpp
    bindings/bind_process_proxy_node.cpp
    bindings/bind_pipeline.cpp
)

nanobind_add_module(visionpipe_python
    NB_STATIC LTO
    ${VISIONPIPE_BINDING_SOURCES}
)

target_link_libraries(visionpipe_python PRIVATE
    visionpipe_core
    visionpipe_nodes
    ${OpenCV_LIBS}
)

target_include_directories(visionpipe_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

if(VISIONPIPE_USE_CUDA)
    target_include_directories(visionpipe_python PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(visionpipe_python PRIVATE VISIONPIPE_USE_CUDA)
endif()

if(VISIONPIPE_USE_TENSORRT)
    target_include_directories(visionpipe_python PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_compile_definitions(visionpipe_python PRIVATE VISIONPIPE_USE_TENSORRT)
endif()

set_target_properties(visionpipe_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)
