#pragma once
extern "C" void alloc_d(long long N, char** buff);

extern "C" void free_d(long long N, char* buff);

extern "C" void init_d(size_t N, float* buff);

extern "C" void compute_d(size_t N, float* d_array, int world_rank);

