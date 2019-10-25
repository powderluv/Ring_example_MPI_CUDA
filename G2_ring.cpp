#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include "util.hpp"

float* allc_G(float* G2);

//void updateG4();

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    if (mpi_size != 2) {
//        std::cout << "Run with two ranks.";
//        MPI_Abort(MPI_COMM_WORLD, -1);
//        exit(-1);
//    }

    int n_recevied = 0;

    std::vector<int> G2s;
    G2s.reserve(mpi_size);

//    for(int i = 0; i<mpi_size; i++)
//    {
////        alloc_d(4, &G2s[i]);
//    }

    G2s[rank] = rank;

    int flag = 0;
    int tag = rank;
    int count;
    MPI_Status status;
    MPI_Request request;

    while(n_recevied <= mpi_size)
    {
        MPI_Isend(&G2s[rank], 1, MPI_INT, (rank + 1) % mpi_size, tag, MPI_COMM_WORLD, &request);

        while(!flag)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }
        MPI_Get_count(&status, MPI_INT, &count);
        MPI_Recv(&G2s[status.MPI_SOURCE], count, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

        std::cout << "I am rank " << rank << " received " << G2s[status.MPI_SOURCE] << " from rank " << status.MPI_TAG << "\n";
        if(status.MPI_TAG != rank)
        {
            MPI_Isend(&G2s[status.MPI_SOURCE], 1, MPI_INT, (rank + 1) % mpi_size, status.MPI_TAG, MPI_COMM_WORLD, &request);

            // updateG4();
        }
        else
        {
            // do nothing
        }

        n_recevied++;
    }
//    updateG4();

    // updateG4 local
    // send myself to right


//    while(n_recevied <= mpi_size - 1)
//    {
//        // if there is a G2 waiting
//        // then receive it, then updateG4
//        // flip the flag in the queue received
//        // send it to my right
//
//        n_recevied++;
//    }

    MPI_Finalize();
}

float* allc_G(float* G2)
{
    alloc_d(4, &G2);
}


//if (rank != 0) {
//MPI_Recv(left, 1, MPI_FLOAT, rank - 1, 0,
//MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//printf("Process %d received token %d from process %d\n",
//rank, 42, rank - 1);
//} else {
//// compute G2 if process is 0
//}
//MPI_Send(G2, 1, MPI_FLOAT, (rank + 1) % mpi_size,
//0, MPI_COMM_WORLD);
//
//// Now process 0 can receive from the last process.
//if (rank == 0) {
//MPI_Recv(left, 1, MPI_FLOAT, mpi_size - 1, 0,
//MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//printf("Process %d received token %d from process %d\n",
//rank, 42, mpi_size - 1);
//}