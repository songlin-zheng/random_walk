cmake_minimum_required(VERSION 3.5)
project(random_walk LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_library(cuda_lib SHARED
            ./helper.cu
            ./rwalk_optimized.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --std=c++11 -lineinfo)
add_compile_definitions(DEBUG=1)
add_compile_definitions(PREPROCESSING=1)

add_executable(${PROJECT_NAME} rwalk_kernel.cc)

set_property(TARGET ${PROJECT_NAME}
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
            cuda_lib
            ${LIBS})
