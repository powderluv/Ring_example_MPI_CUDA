#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include "util.hpp"

void computeG(){}

void updateG4(){}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // each G2 say has 4 elements
    int n_elems = 4;
    float* G2;
    // allocate a sequence of buffers for all G2s
    std::vector<float*> G2s;
    // G2 is currently empty
    alloc_d(n_elems, &G2);
    for(int i = 0; i<mpi_size; i++)
    {
        G2s.emplace_back(G2);
    }

    // generate G2 and fill some value in
    computeG();

    int flag = 0;
    int tag = rank;
    bool sent = false;
    MPI_Status status;
    MPI_Request request;

    int iter = 0;

    // do once at the beginning, sent local G2 to right neighbour
    if (!sent) {
        // each G2 is tagged with its birth rank
        MPI_Isend(G2s[rank], n_elems, MPI_FLOAT, (rank + 1) % mpi_size, rank, MPI_COMM_WORLD, &request);
        std::cout << "Rank " << rank << " sent its G2!" << "\n";
        sent = true;
    }

    while(iter < mpi_size) {
        // probe any available incoming G2
        while (flag == 0) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        }

        // we found one available G2 from left neighbor, let's place it into corresponding buffer position
        if (flag) {
            // birth rank (tag) <-> position of sequence buffer
            MPI_Recv(G2s[status.MPI_TAG], n_elems, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            std::cout << "Rank " << rank << " received G2 [ " << status.MPI_TAG << " ] from rank " << status.MPI_SOURCE << "\n";
            flag = 0;
        }

        if(status.MPI_TAG != rank)
        {
            std::cout << "Rank " << rank << " is sending G2 [ " << status.MPI_TAG << " ] to rank " << (rank + 1) % mpi_size << "\n";
            // forward G2 to my right neighbor using non-blocking Isend
            MPI_Isend(G2s[status.MPI_TAG], 1, MPI_FLOAT, (rank + 1) % mpi_size, status.MPI_TAG, MPI_COMM_WORLD, &request);
        }
        else
        {
            std::cout << "Rank " << rank << " will not send G2 [ " << status.MPI_TAG << " ] to anywhere! \n";
            // this G2 was originally from me, it has travel around the ring and can be retired
            // do nothing
        }

        iter++;
    }

    for(int i = 0; i < mpi_size; i++)
    {
        // every G2 do your work to complete one cell of G4
        updateG4(); // G2s[i];
    }

    // MPI gather to complete full final G4

    MPI_Finalize();
}
