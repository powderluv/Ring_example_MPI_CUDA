#pragma once
extern "C" void compute(int r, int c, float** array, int world_rank);

extern "C" void alloc_d(size_t N, float** buff);

extern "C" void init_d(size_t N, float* buff);

extern "C" void compute_d(size_t N, float* d_array, int world_rank);

