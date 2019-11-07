#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <stdexcept>
#include <array>

#include "device_allocator.hpp"

// Prints an error message containing error, function_name, file_name, line and extra_error_string.
void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string = "");

void print_helper(float* G4, int index);

template <typename ScalarType, class Allocator = dca::linalg::util::DeviceAllocator<ScalarType>>
class ReshapableMatrix {
public:
    using ValueType = ScalarType;
    using ThisType = ReshapableMatrix<ScalarType, Allocator>;
    // Default contructor creates a matrix of zero size and capacity.
    ReshapableMatrix() = default;
    // Initializes a square size x size matrix.
    ReshapableMatrix(int size);
    // Initializes a square size.first x size.second matrix.
    ReshapableMatrix(std::pair<int, int> size);
private:
    static std::size_t nextCapacity(std::size_t size);
    inline static size_t nrElements(std::pair<int, int> size) {
        return static_cast<size_t>(size.first) * static_cast<size_t>(size.second);
    }

    std::pair<int, int> size_ = std::make_pair(0, 0);
    std::size_t capacity_ = 0;

    ValueType* data_ = nullptr;
};

template <typename ScalarType, class Allocator = dca::linalg::util::DeviceAllocator<ScalarType>>
class TpAccumulator
{
public:
    TpAccumulator()
    {}

//    void computeGSingleband(int s);

    void computeG(float* G2, int rank, size_t n_elems)
    {
//        computeGSingleband(0);
        const int n_threads = 512;
        const int n_blocks = (n_elems + (n_threads-1))/ n_threads;
//        __generateG2_in_kernel__<<<n_blocks,n_threads>>>(G2, rank, n_elems);
        return;
    }

    void update_local_G4(float* G2, float* G4, int rank, size_t n_elems)
    {
        int n_blocks = n_elems / 512;
//        __update_local_G4_in_kernel__<<<n_blocks,512>>>(G2, G4, rank, n_elems);
        return;
    }
private:
    using RMatrix =
    ReshapableMatrix<ScalarType, Allocator>;
    std::array<RMatrix, 2> G_;

    bool initialized_ = false;
};

template <typename T>
void CudaMemoryCopy(T* dest, T* src, size_t size) {
    cudaError_t ret = cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    if (ret != cudaSuccess) {
        printErrorMessage(std::string(cudaGetErrorString(ret)), __FUNCTION__, __FILE__, __LINE__, "\t cuda mem copy failed " );
        throw std::logic_error(__FUNCTION__);
    }
}