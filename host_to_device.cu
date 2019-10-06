#include <chrono>
#include <stdio.h>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "host_to_device.hpp"

typedef std::chrono::high_resolution_clock Clock;

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

  auto start_h2d = Clock::now();

  cudaMemcpy(d_array, &(array[0][0]), r*c*sizeof(float), cudaMemcpyHostToDevice);

  auto end_h2d = Clock::now();
  auto time_h2d = std::chrono::duration_cast<std::chrono::duration<double>>(end_h2d - start_h2d).count();
  std::cout << "rank " << world_rank << ": time spent on H2D mem copy is :"<< time_h2d << " seconds \n";

  __compute_in_kernel__<<<1,1>>>(r, c, d_array, world_rank);

  std::cout << "array is computed at rank [" << world_rank << "]'s device.\n";

  auto start_d2h = Clock::now();

  cudaMemcpy(&(array[0][0]), d_array, r*c*sizeof(float), cudaMemcpyDeviceToHost);

  auto end_d2h = Clock::now();
  auto time_d2h = std::chrono::duration_cast<std::chrono::duration<double>>(end_d2h - start_d2h).count();
  std::cout << "rank " << world_rank << ": time spent on d2h mem copy is :"<< time_d2h << " seconds \n";
}

