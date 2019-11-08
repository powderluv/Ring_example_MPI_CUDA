// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Raffaele Solca' (rasolca@itp.phys.ethz.ch)
//
// This file provides memory copy utilities.

#ifndef DCA_LINALG_UTIL_COPY_HPP
#define DCA_LINALG_UTIL_COPY_HPP

#include <cassert>
#include <complex>
#include <cstring>
#include "device_type.hpp"

#include <cuda_runtime.h>
#include "error_cuda.hpp"
#include "stream_functions.hpp"

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

// Fully synchronous 1D memory copy, i.e. all operations in the GPU queue are executed before the
// execution of this copy.
// The host continues the execution of the program when the copy is terminated.
template <typename ScalarType>
void memoryCopy(ScalarType* dest, const ScalarType* src, size_t size) {
cudaError_t ret = cudaMemcpy(dest, src, size * sizeof(ScalarType), cudaMemcpyDefault);
checkRC(ret);
}

// Fully synchronous 2D memory copy, i.e. all operations in the GPU queue are executed before the
// execution of this copy.
// The host continues the execution of the program when the copy is terminated.
// Preconditions: ld_dest >= size.first, ld_src >= size.first.
template <typename ScalarType>
void memoryCopy(ScalarType* dest, int ld_dest, const ScalarType* src, int ld_src,
        std::pair<int, int> size) {
cudaError_t ret = cudaMemcpy2D(dest, ld_dest * sizeof(ScalarType), src, ld_src * sizeof(ScalarType),
                         size.first * sizeof(ScalarType), size.second, cudaMemcpyDefault);
checkRC(ret);
}

// Asynchronous 1D memory copy.
template <typename ScalarType>
void memoryCopyAsync(ScalarType* dest, const ScalarType* src, size_t size, const cudaStream_t stream) {
cudaError_t ret = cudaMemcpyAsync(dest, src, size * sizeof(ScalarType), cudaMemcpyDefault, stream);
checkRC(ret);
}

// Asynchronous 1D memory copy (stream = getStream(thread_id, stream_id)).
// Preconditions: 0 <= thread_id < DCA_MAX_THREADS,
//                0 <= stream_id < DCA_STREAMS_PER_THREADS.
template <typename ScalarType>
void memoryCopyAsync(ScalarType* dest, const ScalarType* src, size_t size, int thread_id,
             int stream_id = 0) {
memoryCopyAsync(dest, src, size, getStream(thread_id, stream_id));
}

// Asynchronous 2D memory copy.
// Preconditions: ld_dest >= size.first, ld_src >= size.firs.
template <typename ScalarType>
void memoryCopyAsync(ScalarType* dest, int ld_dest, const ScalarType* src, int ld_src,
             std::pair<int, int> size, const cudaStream_t stream) {
cudaError_t ret =
cudaMemcpy2DAsync(dest, ld_dest * sizeof(ScalarType), src, ld_src * sizeof(ScalarType),
                size.first * sizeof(ScalarType), size.second, cudaMemcpyDefault, stream);
checkRC(ret);
}

// Asynchronous 2D memory copy (stream = getStream(thread_id, stream_id)).
// Preconditions: ld_dest >= size.first, ld_src >= size.first,
//                0 <= thread_id < DCA_MAX_THREADS,
//                0 <= stream_id < DCA_STREAMS_PER_THREADS.
template <typename ScalarType>
void memoryCopyAsync(ScalarType* dest, int ld_dest, const ScalarType* src, int ld_src,
             std::pair<int, int> size, int thread_id, int stream_id) {
    memoryCopyAsync(dest, ld_dest, src, ld_src, size, getStream(thread_id, stream_id));
}

// Asynchronous 1D memory copy (stream = getStream(thread_id, stream_id))
// + synchronization of stream.
// Preconditions: 0 <= thread_id < DCA_MAX_THREADS,
template <typename ScalarType>
void memoryCopy(ScalarType* dest, const ScalarType* src, size_t size, int thread_id, int stream_id) {
    memoryCopyAsync(dest, src, size, thread_id, stream_id);
    syncStream(thread_id, stream_id);
}

// Asynchronous 2D memory copy (stream = getStream(thread_id, stream_id))
// + synchronization of stream.
// Preconditions: ld_dest >= size.first, ld_src >= size.first,
//                0 <= thread_id < DCA_MAX_THREADS,
//                0 <= stream_id < DCA_STREAMS_PER_THREADS.
template <typename ScalarType>
void memoryCopy(ScalarType* dest, int ld_dest, const ScalarType* src, int ld_src,
        std::pair<int, int> size, int thread_id, int stream_id) {
    memoryCopyAsync(dest, ld_dest, src, ld_src, size, thread_id, stream_id);
    syncStream(thread_id, stream_id);
}
}  // util
}  // linalg
}  // dca

#endif  // DCA_LINALG_UTIL_COPY_HPP
