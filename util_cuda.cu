#include "util.hpp"

#include <chrono>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sstream>

typedef std::chrono::high_resolution_clock Clock;

__global__
void __init_in_kernel__(long long N, char* d_array, char c)
{
  // do some computation on the device
  for(int i = 0; i<N; i++)
  {
      d_array[i] = c;
  }
}

void alloc_d(long long N, float ** buff)
{
	cudaMalloc((void**)buff, N * sizeof(float));
}

void alloc_d_char(long long N, char ** buff)
{
    cudaMalloc((void**)buff, N * sizeof(char));
}

void free_d(char* buff)
{
    cudaFree(buff);
}

void init_d(long long N, char* buff, char c)
{
    __init_in_kernel__<<<1,1>>>(N, buff, c);
}

void printErrorMessage(std::string error, std::string function_name, std::string file_name,
                       int line, std::string extra_error_string) {
    std::stringstream s;

    s << "Error in function: " << function_name;
    s << " (" << file_name << ":" << line << ")" << std::endl;
    s << "The function returned: " << error << std::endl;
    if (extra_error_string != "")
        s << extra_error_string << std::endl;

    std::cout << s.str() << std::endl;
}
