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
      -std=c++11
      -Xcompiler -Wall
      -Xcompiler -Wextra
      -Xcompiler -Wno-unused-parameter
      -Xcompiler -Wno-switch)

CUDA_ADD_LIBRARY(host_to_device host_to_device.cu)

add_executable(main main.cpp allocation.cpp) 
target_include_directories(main PRIVATE ${CUDA_TOOLKIT_INCLUDE} allocation.hpp)

target_link_libraries(main PRIVATE host_to_device)

