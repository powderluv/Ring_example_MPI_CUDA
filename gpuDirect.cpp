// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Example using MPI_Send and MPI_Recv to pass a message around in a ring.
//
#include <assert.h>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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

    char* s_array;
    char* r_array;

    std::vector<char> msg;
    std::ofstream out("gpuDirect_ping_pong_timings.txt");

    if(rank == 0)
    {
        std::cout << "Size \t time per iter \t bandwitdth [GB/s] \n";
        out << "Size \t time per iter \t bandwitdth [GB/s] \n";
    }

    for (long long size : std::vector<long long>{1, 2, 10, 100, 1000, 1000000, 100000000, 1000000000}) {
        alloc_d(size, &s_array);
        alloc_d(size, &r_array);

        bool ping = 0;
        startTimer();
        for (int i = 0; i < times; ++i) {
            if (rank == ping)
                MPI_Send(s_array, size, MPI_CHAR, !ping, 0, MPI_COMM_WORLD);
            else
                MPI_Recv(r_array, size, MPI_CHAR, ping, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

            ping = !ping;
        }
        auto time = endTimer() / times;

        if(rank == 0) {
            out << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
            std::cout << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
        }

        free_d(size, s_array);
        free_d(size, r_array);
    }

    MPI_Finalize();
}
