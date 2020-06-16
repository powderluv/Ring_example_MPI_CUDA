#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include "util_cuda.hpp"
#include "util_mpi.hpp"

#define MOD(x,n) ((x) % (n))

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, mpi_size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // sync all processors at the beginning
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_Request recv_request;
    MPI_Request send_request;
    MPI_Status status;

    int left_neighbor = MOD((rank-1 + mpi_size), mpi_size);
    int right_neighbor = MOD((rank+1 + mpi_size), mpi_size);
    const bool is_even = rank % 2;

    // number of G2s
    int niter = 1000;

    size_t n_elems = 8388608;

    float* G2_h = (float*)malloc(n_elems * sizeof(float));
    for(int i = 0; i < n_elems; i++)
    {
        G2_h[i] = (double) rank;
    }

    float* G2 = nullptr;
    float* G4 = nullptr;
    float* sendbuff_G2 = nullptr;
    float* recvbuff_G2 = nullptr;

    cudaMalloc((void**)&G2, n_elems * sizeof(float));
    cudaMalloc((void**)&G4, n_elems * sizeof(float));
    cudaMalloc((void**)&sendbuff_G2, n_elems * sizeof(float));
    cudaMalloc((void**)&recvbuff_G2, n_elems * sizeof(float));

    cudaMemcpy(G2, G2_h, n_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(G4, G2_h, n_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sendbuff_G2, G2_h, n_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(recvbuff_G2, G2_h, n_elems * sizeof(float), cudaMemcpyHostToDevice);

    double start_time, end_time;
    // sync all processors at the end
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }
    for(int i = 0; i < niter; i++)
    {
        for(int icount=0; icount < (mpi_size-1); icount++)
        {
            if(is_even) {
                MPI_CHECK(MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, 1, MPI_COMM_WORLD, &recv_request));
                MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, 1, MPI_COMM_WORLD, &send_request));
            }
            else
            {
                MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, 1, MPI_COMM_WORLD, &send_request));
                MPI_CHECK(MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, 1, MPI_COMM_WORLD, &recv_request));
            }

            MPI_CHECK(MPI_Wait(&recv_request, &status));
            MPI_CHECK(MPI_Wait(&send_request, &status));
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
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
