#include <chrono>
#include <stdio.h>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "host_to_device.hpp"

typedef std::chrono::high_resolution_clock Clock;

__global__
void __compute_in_kernel__(size_t N, float* d_array, int world_rank)
{
  // do some computation on the device
  for(int i = 0; i<N; i++)
  {
      d_array[i] += 1;
	printf("%f, ", d_array[i]);
  }
	printf("\nrank: %d \n", world_rank);
}

__global__
void __init_in_kernel__(size_t N, float* d_array)
{
    for(int i = 0; i<N; i++)
    {
        d_array[i] = 42.0;
    }
}

void alloc_d(long long N, char** buff)
{
	cudaMalloc((void**)buff, N * sizeof(char));
}

void free_d(long long N, char* buff)
{
    cudaFree(buff);
}

void init_d(size_t N, float* buff)
{
    __init_in_kernel__<<<1,1>>>(N, buff);
}

void compute_d(size_t N, float* buff, int world_rank)
{
	  __compute_in_kernel__<<<1,1>>>(N, buff, world_rank);
}
