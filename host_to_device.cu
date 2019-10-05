#include <stdio.h>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "host_to_device.hpp"

__global__
void __compute_in_kernel__(int r, int c, float* d_array, int world_rank)
{
  // do some computation on the device
  for(int i = 0; i<r*c; i++)
  {
	d_array[i] += 1;
	assert((float)(world_rank+1+i)==d_array[i]);
  }  
}

void compute(int r, int c, float** array, int world_rank)
{
  float *d_array;

  cudaMalloc((void **)&d_array, r*c*sizeof(float));
  cudaMemcpy(d_array, &(array[0][0]), r*c*sizeof(float), cudaMemcpyHostToDevice);
  
  __compute_in_kernel__<<<1,1>>>(r, c, d_array, world_rank);

  std::cout << "array is computed at rank [" << world_rank << "]'s device.\n";

  cudaMemcpy(&(array[0][0]), d_array, r*c*sizeof(float), cudaMemcpyDeviceToHost);

}

