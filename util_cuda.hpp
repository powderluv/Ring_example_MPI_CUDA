#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void alloc_d(long long N, float** buff);

extern "C" void alloc_d_char(long long N, char** buff);

extern "C" void free_d(char* buff);

extern "C" void init_d(long long N, char* buff, char c);

// Prints an error message containing error, function_name, file_name, line and extra_error_string.
void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string = "");

template<typename T>
T* allocate_on_device(std::size_t n) {
    if (n == 0)
        return nullptr;
    T* ptr;
    cudaError_t ret = cudaMalloc((void**)&ptr, n * sizeof(T));
    if (ret != cudaSuccess) {
        printErrorMessage(std::string(cudaGetErrorString(ret)), __FUNCTION__, __FILE__, __LINE__,
                          "\t DEVICE size requested : " + std::to_string(n * sizeof(T)));
        throw(std::bad_alloc());
    }
    return ptr;
}
