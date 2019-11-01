#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include "util_cuda.hpp"
#include "util_mpi.hpp"

#define MOD(x,n) ((x) % (n))

void generateG2(float* G2){}

void update_local_G4(float* G2){}

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

    // number of G2s
    int niter = 10;

    size_t n_elems = 8000000;
    float* G2 = nullptr;
    float* sendbuff_G2 = nullptr;
    float* recvbuff_G2 = nullptr;

    G2 = allocate_on_device<float>(n_elems);
    sendbuff_G2 = allocate_on_device<float>(n_elems);
    recvbuff_G2 = allocate_on_device<float>(n_elems);

    double start_time, end_time;
    for(int i = 0; i < niter; i++)
    {
        // generate G2 and fill some value in
        generateG2(G2);
        update_local_G4(G2);
        // get ready for send
        CudaMemoryCopy(sendbuff_G2, G2, n_elems);
        int send_tag = 1 + rank;
        send_tag = 1 + MOD(send_tag-1, MPI_TAG_UB); // just to be safe, MPI_TAG_UB is largest tag value

        if (rank == 0)
        {
            start_time = MPI_Wtime();
            printf("start_time: %lf in iteration %d \n", start_time, i);
        }

        for(int icount=0; icount < (mpi_size-1); icount++)
        {
            // encode the originator rank in the message tag as tag = 1 + originator_irank
            int originator_irank = MOD(((rank-1)-icount + 2*mpi_size), mpi_size);
            int recv_tag = 1 + originator_irank;
            recv_tag = 1 + MOD(recv_tag-1, MPI_TAG_UB); // just to be safe, then 1 <= tag <= MPI_TAG_UB
#ifdef PRINT_DEBUG_INFO
            printf("rank %d receive G2 [%d] from rank %d in iteration %d \n", rank, recv_tag, left_neighbor, i);
#endif
            MPI_CHECK(MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, recv_tag, MPI_COMM_WORLD, &recv_request));

#ifdef PRINT_DEBUG_INFO
            printf("rank %d send G2 [%d] to rank %d  in iteration %d \n", rank, send_tag, right_neighbor, i);
#endif
            MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, send_tag, MPI_COMM_WORLD, &send_request));

            MPI_CHECK(MPI_Wait(&recv_request, &status));
            CudaMemoryCopy(G2, recvbuff_G2, n_elems);
            update_local_G4(G2);
            MPI_CHECK(MPI_Wait(&send_request, &status)); // wait for sendbuf_G2 to be available again

            // get ready for send
            CudaMemoryCopy(G2, sendbuff_G2, n_elems);
            send_tag = recv_tag;
        }

        if (rank == 0)
        {
            end_time = MPI_Wtime();
            printf("end_time: %lf in iteration %d \n", end_time, i);
            printf("G2 has traveled across the world, time spent: %lf in iteration %d \n", end_time - start_time, i);
        }
    }

    // sync all processors at the end
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_CHECK(MPI_Finalize());
}
