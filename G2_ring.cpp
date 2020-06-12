#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <future>

#include "util_cuda.hpp"
#include "util_mpi.hpp"

#define MOD(x,n) ((x) % (n))

void perform_one_communication_step(const int left_neighbor, const int right_neighbor, const int rank,
                                     float* G2, float* sendbuff_G2, float* recvbuff_G2, float* G4, size_t  n_elems, int thread_id)
{
    MPI_Request recv_request;
    MPI_Request send_request;
    MPI_Status status;

    MPI_CHECK(MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, thread_id, MPI_COMM_WORLD, &recv_request));
    MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, thread_id, MPI_COMM_WORLD, &send_request));

    MPI_CHECK(MPI_Wait(&recv_request, &status));
    CudaMemoryCopy(G2, recvbuff_G2, n_elems);
    update_local_G4(G2, G4, rank, n_elems);
    MPI_CHECK(MPI_Wait(&send_request, &status)); // wait for sendbuf_G2 to be available again

    // get ready for send
    CudaMemoryCopy(sendbuff_G2, G2, n_elems);
}

void task(size_t n_elems, int niter, int thread_id) {
    int rank, mpi_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // sync all processors at the beginning
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int left_neighbor = MOD((rank-1 + mpi_size), mpi_size);
    int right_neighbor = MOD((rank+1 + mpi_size), mpi_size);

    float* G2 = nullptr;
    float* G4 = nullptr;
    float* sendbuff_G2 = nullptr;
    float* recvbuff_G2 = nullptr;

    G2 = allocate_on_device<float>(n_elems);
    G4 = allocate_on_device<float>(n_elems);
    sendbuff_G2 = allocate_on_device<float>(n_elems);
    recvbuff_G2 = allocate_on_device<float>(n_elems);

    for(int i = 0; i < niter; i++)
    {
        // generate G2 and fill some value in
        generateG2(G2, rank, n_elems);
        update_local_G4(G2, G4, rank, n_elems);

        // get ready for send
        CudaMemoryCopy(sendbuff_G2, G2, n_elems);

        for(int icount=0; icount < (mpi_size-1); icount++)
        {
            perform_one_communication_step(left_neighbor, right_neighbor, rank,
                                                G2, sendbuff_G2, recvbuff_G2, G4, n_elems, thread_id);
        }
    }
}

int main(int argc, char **argv) {
    int is_initialized_ = -1;
    MPI_Initialized(&is_initialized_);
    if (!is_initialized_)
    {
        int provided = 0;
        constexpr int required = MPI_THREAD_FUNNELED;
        MPI_Init_thread(&argc, &argv, required, &provided);
        if (provided < required)
            throw(std::logic_error("MPI does not provide adequate thread support."));
    }

    int rank, mpi_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    double start_time, end_time;

    size_t n_elems = 26214400; // 2 ^ 23
    int niter = 10;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }

    // start thread pool
    std::vector<std::future<void> > pool;

    for(int thread_id = 0; thread_id < 7; thread_id++)
    {
        pool.emplace_back(std::async(task, n_elems, niter, thread_id));
    }

    for(auto& t: pool)
    {
        t.get();
    }
    // end thread pool

    if (rank == 0)
    {
        end_time = MPI_Wtime();
        printf("Total time spent on %d iteration: %lf \n", niter, (end_time - start_time));
        printf("Number of ranks: %d, number of float elements %d \n", mpi_size, n_elems);
        printf("Average time spent on 1 iteration: %lf \n", (end_time - start_time) / niter);
    }

    // sync all processors at the end
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_CHECK(MPI_Finalize());
}