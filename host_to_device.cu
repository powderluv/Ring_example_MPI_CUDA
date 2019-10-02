#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "host_to_device.hpp"

__global__
void __compute_in_kernel__(int N, int* d_array, int world_rank)
{
  // do some computation on the device
  for(int i = 0; i<N; i++)
  	d_array[i] += 1;
  
  printf("array is computed at rank[%d], array[0] is %d, printing from rank %d's device\n", world_rank, d_array[0], world_rank);
}

extern "C" void compute(int N, int* array, int world_rank)
{
  int *d_array;

  cudaMalloc(&d_array, N*sizeof(int));
  cudaMemcpy(d_array, array, N*sizeof(int), cudaMemcpyHostToDevice);
  
  __compute_in_kernel__<<<1,1>>>(N, d_array, world_rank);

  cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);

}

