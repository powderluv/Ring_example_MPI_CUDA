cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

CUDA_ADD_LIBRARY(tp_accumulator_gpu SHARED tp_accumulator_kernels.cu)

add_library(cuda_utils STATIC error_cuda.cpp)
target_include_directories(cuda_utils PUBLIC ${CUDA_TOOLKIT_INCLUDE})

add_executable(main_ring main_ring.cpp)
target_include_directories(main_ring PUBLIC ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(main_ring PUBLIC tp_accumulator_gpu cuda_utils)
