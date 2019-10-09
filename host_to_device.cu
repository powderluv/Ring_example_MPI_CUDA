#include <chrono>
#include <stdio.h>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "host_to_device.hpp"

typedef std::chrono::high_resolution_clock Clock;

__global__
void __compute_in_kernel__(int r, int c, char* d_array, int world_rank)
{
  // do some computation on the device
  for(int i = 0; i<r*c; i++)
  {
	printf("%d, ", d_array[i]);
  }  
	printf("\n");
}

void d2d_alloc(int r, int c, char** buff)
{
	cudaMalloc((void**)buff, (size_t)(r*c));
}

void d2d_memset(int r, int c, void* buff)
{
	  cudaMemset(buff, 'b', r*c);
}

void d2d_compute(int r, int c, char* buff)
{
	  __compute_in_kernel__<<<1,1>>>(r, c, buff, 1);
}
