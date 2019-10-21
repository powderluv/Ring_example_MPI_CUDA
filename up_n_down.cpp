#include <assert.h>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "allocation.hpp"
#include "host_to_device.hpp"
#include "timer.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_size != 2) {
        std::cout << "Run with two ranks.";
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }

    const int times = 100;

    char* s_d_array;
    char* r_d_array;

    char* s_h_array;
    char* r_h_array;

    std::vector<char> msg;
    std::ofstream out("gpuDirect_ping_pong_timings.txt");

    if(rank == 0)
    {
        std::cout << "Size \t time per iter \t bandwitdth [GB/s] \n";
        out << "Size \t time per iter \t bandwitdth [GB/s] \n";
    }

    for (long long size : std::vector<long long>{1, 2, 10, 100, 1000, 1000000, 30000000, 100000000, 1000000000}) {
        alloc_d(size, &s_d_array);
        init_d(size, s_d_array, 'a');
        alloc_d(size, &r_d_array);

        s_h_array = (char*) malloc(size * sizeof(char));
        r_h_array = (char*) malloc(size * sizeof(char));

        bool ping = 0;
        startTimer();
        for (int i = 0; i < times; ++i) {
            if (rank == ping)
            {
                cudaMemcpy(s_h_array, s_d_array, size, cudaMemcpyDeviceToHost);
                MPI_Send(s_h_array, size, MPI_CHAR, !ping, 0, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Recv(r_h_array, size, MPI_CHAR, ping, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                cudaMemcpy(r_d_array, r_h_array, size, cudaMemcpyHostToDevice);
            }
            ping = !ping;
        }
        auto time = endTimer() / times;

        if(rank == 0) {
            out << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
            std::cout << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
        }

        free_d(s_d_array);
        free_d(r_d_array);
        free(s_h_array);
        free(r_h_array);
    }

    MPI_Finalize();
}
