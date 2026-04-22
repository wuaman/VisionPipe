# Resolve TensorRT for VisionPipe:
# 1) Prefer CMake config (TensorRTConfig.cmake), e.g. some SDK / conda layouts.
# 2) Otherwise locate NvInfer.h + libnvinfer (+ optional libnvonnxparser) for tar/apt installs
#    that ship no CMake package (common for NVIDIA Linux tar and Debian libnvinfer-dev).

set(TENSORRT_ROOT "" CACHE PATH
    "TensorRT root: directory containing include/ and lib/ (e.g. extracted TensorRT-10.x tarball).")
set(VISIONPIPE_TRT_LIB_DIR "" CACHE PATH
    "TensorRT lib directory only (e.g. .../TensorRT-x/lib). include/ is assumed as sibling ../include.")

find_package(TensorRT 8.6 QUIET CONFIG)

if(TensorRT_FOUND)
    message(STATUS "TensorRT: using CMake config (TensorRT_DIR=${TensorRT_DIR})")
else()
    message(STATUS "TensorRT: no CMake config found; probing NvInfer.h and shared libraries.")

    set(_vp_trt_inc_hints)
    set(_vp_trt_lib_hints)

    if(TENSORRT_ROOT)
        list(APPEND _vp_trt_inc_hints
            "${TENSORRT_ROOT}/include"
            "${TENSORRT_ROOT}/include/x86_64-linux-gnu"
            "${TENSORRT_ROOT}/include/aarch64-linux-gnu")
        list(APPEND _vp_trt_lib_hints
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib64"
            "${TENSORRT_ROOT}/lib/stubs")
    endif()

    if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
        list(APPEND _vp_trt_inc_hints
            "$ENV{TENSORRT_ROOT}/include"
            "$ENV{TENSORRT_ROOT}/include/x86_64-linux-gnu"
            "$ENV{TENSORRT_ROOT}/include/aarch64-linux-gnu")
        list(APPEND _vp_trt_lib_hints
            "$ENV{TENSORRT_ROOT}/lib"
            "$ENV{TENSORRT_ROOT}/lib64"
            "$ENV{TENSORRT_ROOT}/lib/stubs")
    endif()

    if(VISIONPIPE_TRT_LIB_DIR)
        list(APPEND _vp_trt_lib_hints "${VISIONPIPE_TRT_LIB_DIR}")
        get_filename_component(_vp_trt_from_lib "${VISIONPIPE_TRT_LIB_DIR}" DIRECTORY)
        list(APPEND _vp_trt_inc_hints
            "${_vp_trt_from_lib}/include"
            "${_vp_trt_from_lib}/include/x86_64-linux-gnu"
            "${_vp_trt_from_lib}/include/aarch64-linux-gnu")
    endif()

    list(APPEND _vp_trt_lib_hints
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib)
    list(APPEND _vp_trt_inc_hints
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu
        /usr/include)

    find_path(TensorRT_INCLUDE_DIR NvInfer.h HINTS ${_vp_trt_inc_hints})

    find_library(TensorRT_NVINFER_LIBRARY NAMES nvinfer HINTS ${_vp_trt_lib_hints})
    find_library(TensorRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser HINTS ${_vp_trt_lib_hints})

    if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIBRARY)
        message(FATAL_ERROR
            "TensorRT not found.\n"
            "  - If you have TensorRTConfig.cmake, set TensorRT_DIR to .../lib/cmake/TensorRT, or add its prefix to CMAKE_PREFIX_PATH.\n"
            "  - For NVIDIA tar (bin/include/lib only) or apt dev packages without CMake config, set either:\n"
            "      -DTENSORRT_ROOT=/path/to/TensorRT-10.x   (directory that contains include/ and lib/)\n"
            "    or -DVISIONPIPE_TRT_LIB_DIR=/path/to/lib   (and include/ as sibling ../include)\n"
            "    or export TENSORRT_ROOT=/path/to/TensorRT-10.x\n"
            "  Missing: TensorRT_INCLUDE_DIR='${TensorRT_INCLUDE_DIR}' TensorRT_NVINFER_LIBRARY='${TensorRT_NVINFER_LIBRARY}'")
    endif()

    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    set(TensorRT_LIBRARIES "${TensorRT_NVINFER_LIBRARY}")
    if(TensorRT_NVONNXPARSER_LIBRARY)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_NVONNXPARSER_LIBRARY}")
    endif()
    set(TensorRT_FOUND TRUE)
    message(STATUS "TensorRT: include dirs: ${TensorRT_INCLUDE_DIRS}")
    message(STATUS "TensorRT: libraries: ${TensorRT_LIBRARIES}")
endif()
