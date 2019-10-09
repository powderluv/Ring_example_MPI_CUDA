#pragma once
extern "C" void compute(int r, int c, float** array, int world_rank);

extern "C" void d2d_alloc(int r, int c, char** buff);

extern "C" void d2d_memset(int r, int c, void* buff);

extern "C" void d2d_compute(int r, int c, char* d_array);

