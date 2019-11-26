cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mpi_cuda)

# Find CUDA.
find_package(CUDA REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

set(CUDA_GPU_ARCH "sm_60" CACHE STRING "Name of the real architecture to build for.")

include_directories(${CUDA_TOOLKIT_INCLUDE})

CUDA_INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

set(CUDA_SEPARABLE_COMPILATION ON)

list(APPEND CUDA_NVCC_FLAGS
      -arch=${CUDA_GPU_ARCH}
      -std=c++14
      -Xcompiler -Wall
      -Xcompiler -Wextra
      -Xcompiler -Wno-unused-parameter
      -Xcompiler -Wno-switch)

CUDA_ADD_LIBRARY(util SHARED util.cu)

add_executable(gpuDirect gpuDirect.cpp allocation.cpp allocation.hpp)
target_include_directories(gpuDirect PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(gpuDirect PRIVATE util)

add_executable(up_n_down up_n_down.cpp allocation.hpp)
target_include_directories(up_n_down PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(up_n_down PRIVATE util)

add_executable(cuda_device_list cuda_device_list.cpp)
target_include_directories(cuda_device_list PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(cuda_device_list ${CUDA_LIBRARIES})

add_executable(G2_ring G2_ring.cpp)
target_include_directories(G2_ring PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(G2_ring PRIVATE util)

add_subdirectory(tests)
