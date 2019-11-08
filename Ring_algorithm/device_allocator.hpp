//
// Created by Wei, Weile on 11/7/19.
//

#ifndef MPI_CUDA_DEVICE_ALLOCATOR_HPP
#define MPI_CUDA_DEVICE_ALLOCATOR_HPP

#include "device_type.hpp"
#include "error_cuda.hpp"

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

template <typename T, DeviceType device_name = dca::linalg::GPU>
class DeviceAllocator {
protected:
    T* allocate(std::size_t n) {
        if (n == 0)
            return nullptr;
        T* ptr;
        cudaError_t ret = cudaMalloc((void**)&ptr, n * sizeof(T));
        if (ret != cudaSuccess) {
            printErrorMessage(ret, __FUNCTION__, __FILE__, __LINE__,
                              "\t DEVICE size requested : " + std::to_string(n * sizeof(T)));
            throw(std::bad_alloc());
        }
        return ptr;
    }

    void deallocate(T*& ptr, std::size_t /*n*/ = 0) noexcept {
        cudaError_t ret = cudaFree(ptr);
        if (ret != cudaSuccess) {
            printErrorMessage(ret, __FUNCTION__, __FILE__, __LINE__);
            std::terminate();
        }
        ptr = nullptr;
    }

public:
    // SFINAE method for setting managed memory stream.
    void setStream(const cudaStream_t /*stream*/) const {}
};

}  // util
}  // linalg
}  // dca

#endif //MPI_CUDA_DEVICE_ALLOCATOR_HPP
