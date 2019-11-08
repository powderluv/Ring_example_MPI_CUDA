#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <stdexcept>
#include <array>

#include "device_allocator.hpp"
#include "reshapable_matrix.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace accumulator {
// dca::phys::solver::accumulator::

template <typename ScalarType, class Allocator = dca::linalg::util::DeviceAllocator<ScalarType>>
class TpAccumulator;

template <typename ScalarType>
class TpAccumulator<ScalarType>
{
private:
    void computeGSingleband(int s);
public:
    TpAccumulator()
    {
//        std::cout << "I am constructed\n";
    }

    void computeG(float* G2, int rank, size_t n_elems)
    {
        computeGSingleband(0);
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
    dca::linalg::ReshapableMatrix<ScalarType>;
    std::array<RMatrix, 2> G_;

    bool initialized_ = false;
};

template <typename ScalarType>
void TpAccumulator<ScalarType>::computeGSingleband(const int s) {
//    details::computeGSingleband(G_[s].ptr(), G_[s].leadingDimension(), get_G0()[s].ptr(),
//                                KDmn::dmn_size(), n_pos_frqs_, beta_, streams_[s]);
//    assert(cudaPeekAtLastError() == cudaSuccess);
    std::cout << "hello \n";
}

}  // namespace accumulator
}  // namespace solver
}  // namespace phys
}  // namespace dca




// Prints an error message containing error, function_name, file_name, line and extra_error_string.
void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string = "");

void print_helper(float* G4, int index);

template <typename T>
void CudaMemoryCopy(T* dest, T* src, size_t size) {
    cudaError_t ret = cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
    if (ret != cudaSuccess) {
        printErrorMessage(std::string(cudaGetErrorString(ret)), __FUNCTION__, __FILE__, __LINE__, "\t cuda mem copy failed " );
        throw std::logic_error(__FUNCTION__);
    }
}