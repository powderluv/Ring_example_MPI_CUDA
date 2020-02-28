#include <assert.h>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "allocation.hpp"
#include "timer.hpp"
#include "util_cuda.hpp"
#include "util_mpi.hpp"

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/mpi.hpp>
#include <hpx/lcos/future.hpp>

int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, mpi_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (mpi_size != 2) {
        std::cout << "Run with two ranks.";
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, -1));
        exit(-1);
    }

hpx::mpi::enable_user_polling enable_polling;
hpx::mpi::executor exec(MPI_COMM_WORLD);

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

    for (long long size : std::vector<long long>{1, 2, 10, 100, 1000, 1000000, 30000000, 100000000, 1000000000}) {
        alloc_d_char(size, &s_array);
        init_d(size, s_array, 'a');
        alloc_d_char(size, &r_array);

        bool ping = 0;
        startTimer();
        for (int i = 0; i < times; ++i) {
            if (rank == ping)
	    {
		hpx::future<int> f_send = hpx::async(exec, MPI_Isend, s_array, size, MPI_CHAR, !ping, 0);
		f_send.get();
            }
	    else
	    {
		hpx::future<int> f_recv = hpx::async(exec, MPI_Irecv, r_array, size, MPI_CHAR, ping, 0);
                //MPI_CHECK(MPI_Recv(r_array, size, MPI_CHAR, ping, 0, MPI_COMM_WORLD,
                //         MPI_STATUS_IGNORE));
                f_recv.get();
	    }
            ping = !ping;
        }
        auto time = endTimer() / times;

        if(rank == 0) {
            out << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
            std::cout << size << "\t" << time << "\t" << size / time * 1e-9 << "\n";
        }

        free_d(s_array);
        free_d(r_array);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CHECK(MPI_Finalize());
}
