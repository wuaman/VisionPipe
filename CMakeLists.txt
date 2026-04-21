cmake_minimum_required(VERSION 3.20)

project(VisionPipe
    VERSION 0.1.0
    DESCRIPTION "Video AI Inference Framework with GPU Acceleration"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VISIONPIPE_BUILD_TESTS "Build unit tests" ON)
option(VISIONPIPE_BUILD_PYTHON "Build Python bindings" ON)
option(VISIONPIPE_USE_CUDA "Enable CUDA support" ON)
option(VISIONPIPE_USE_TENSORRT "Enable TensorRT support" ON)

if(VISIONPIPE_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA 11.8 REQUIRED)
    set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
endif()

if(VISIONPIPE_USE_TENSORRT)
    find_package(TensorRT 8.6 REQUIRED)
endif()

find_package(OpenCV 4.7 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(spdlog nlohmann_json)

if(VISIONPIPE_BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

add_subdirectory(src/core)
add_subdirectory(src/hal)

if(VISIONPIPE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(VISIONPIPE_BUILD_PYTHON)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.0.0
    )
    FetchContent_MakeAvailable(nanobind)
    add_subdirectory(python)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")