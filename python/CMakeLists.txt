# VisionPipe Python Bindings
# nanobind 绑定：将 C++ 核心类暴露给 Python

# 收集绑定源文件
set(VISIONPIPE_BINDING_SOURCES
    # 绑定文件将在后续任务中添加
    # bindings/bind_pipeline.cpp
    # bindings/bind_nodes.cpp
    # bindings/bind_frame.cpp
    # bindings/bind_exceptions.cpp
)

# 如果有源文件，创建 Python 模块
if(VISIONPIPE_BINDING_SOURCES)
    nanobind_add_module(visionpipe_python
        NB_STATIC STABLE_ABI LTO
        ${VISIONPIPE_BINDING_SOURCES}
    )

    target_link_libraries(visionpipe_python PRIVATE
        visionpipe_core
        visionpipe_hal
    )

    target_include_directories(visionpipe_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    # 设置输出目录
    set_target_properties(visionpipe_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
else()
    # 暂时创建一个占位模块，确保构建通过
    nanobind_add_module(visionpipe_python
        NB_STATIC STABLE_ABI LTO
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings/placeholder.cpp
    )

    target_link_libraries(visionpipe_python PRIVATE
        visionpipe_core
        visionpipe_hal
    )

    set_target_properties(visionpipe_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
endif()
