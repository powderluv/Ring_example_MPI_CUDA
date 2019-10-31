#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

#include "util.hpp"

#define MOD(x,n) ((x) % (n))

void generateG2(float* G2){}

void update_local_G4(float* G2){}

void copy_d(float* origin, float* copy) {}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request recv_request;
    MPI_Request send_request;
    MPI_Status status;

    int left_neighbor = MOD((rank-1 + mpi_size), mpi_size);
    int right_neighbor = MOD((rank+1 + mpi_size), mpi_size);

    int niter = 1;

    // each G2 say has 4 elements
    int n_elems = 4;
    float* G2;
    float* sendbuff_G2;
    float* recvbuff_G2;
    alloc_d(n_elems, &G2);
    alloc_d(n_elems, &sendbuff_G2);
    alloc_d(n_elems, &recvbuff_G2);


    for(int i = 0; i < niter; i++)
    {
        // generate G2 and fill some value in
        generateG2(G2);
        update_local_G4(G2);
        // get ready for send
        copy_d(G2, sendbuff_G2); // copy into buffer
        int send_tag = 1 + rank;
        send_tag = 1 + MOD(send_tag-1, MPI_TAG_UB); // just to be safe, MPI_TAG_UB is largest tag value

        for(int icount=0; icount < (mpi_size-1); icount++)
        {
            // encode the originator rank in the message tag as tag = 1 + originator_irank
            int originator_irank = MOD(((rank-1)-icount + 2*mpi_size), mpi_size);
            int recv_tag = 1 + originator_irank;
            recv_tag = 1 + MOD(recv_tag-1, MPI_TAG_UB); // just to be safe, then 1 <= tag <= MPI_TAG_UB

            printf("rank %d receive G2 [%d] from rank %d \n", rank, recv_tag, left_neighbor);
            MPI_Irecv(recvbuff_G2, n_elems, MPI_FLOAT, left_neighbor, recv_tag, MPI_COMM_WORLD, &recv_request);

            printf("rank %d send G2 [%d] to rank %d \n", rank, send_tag, right_neighbor);
            MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, send_tag, MPI_COMM_WORLD, &send_request);

            MPI_Wait(&recv_request, &status);
            //G2 = recvbuf_G2; // copy from buffer
            update_local_G4(G2);
            MPI_Wait(&send_request, &status); // wait for sendbuf_G2 to be available again

            // get ready for send
            //sendbuf_G2 = G2;
            send_tag = recv_tag;
        }
    }
    MPI_Finalize();
}

//    int flag = 0;
//    int tag = rank;
//    bool sent = false;
//    MPI_Status status;
//    MPI_Request request;
//
//    int iter = 0;
//
//    // do once at the beginning, sent local G2 to right neighbour
//    if (!sent) {
//        // each G2 is tagged with its birth rank
//        MPI_Isend(G2s[rank], n_elems, MPI_FLOAT, (rank + 1) % mpi_size, rank, MPI_COMM_WORLD, &request);
//        std::cout << "Rank " << rank << " sent its G2!" << "\n";
//        sent = true;
//    }
//
//    while(iter < mpi_size) {
//        // probe any available incoming G2
//        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
//
//        // we found one available G2 from left neighbor, let's place it into corresponding buffer position
//        if (flag) {
//            // birth rank (tag) <-> position of sequence buffer
//            MPI_Recv(G2s[status.MPI_TAG], n_elems, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//            std::cout << "Rank " << rank << " received G2 [ " << status.MPI_TAG << " ] from rank " << status.MPI_SOURCE << "\n";
//
//            flag = 0;
//
//            iter++;
//
//            if(status.MPI_TAG != rank)
//            {
//                std::cout << "Rank " << rank << " is sending G2 [ " << status.MPI_TAG << " ] to rank " << (rank + 1) % mpi_size << "\n";
//                // forward G2 to my right neighbor using non-blocking Isend
//                MPI_Isend(G2s[status.MPI_TAG], 1, MPI_FLOAT, (rank + 1) % mpi_size, status.MPI_TAG, MPI_COMM_WORLD, &request);
//            }
//            else
//            {
//                std::cout << "Rank " << rank << " will not send G2 [ " << status.MPI_TAG << " ] to anywhere! \n";
//                // this G2 was originally from me, it has travel around the ring and can be retired
//                // do nothing
//            }
//        }
//    }
//
//    for(int i = 0; i < mpi_size; i++)
//    {
//        // every G2 do your work to complete one cell of G4
//        updateG4(); // G2s[i];
//    }

    // MPI gather to complete full final G4


