// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Example using MPI_Send and MPI_Recv to pass a message around in a ring.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "allocation.hpp"
#include "host_to_device.hpp"
#include "timer.hpp"

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  float** array;
  int r = 100;
  int c = 100;
  // Receive from the lower process and send to the higher process. Take care
  // of the special case when you are the first process to prevent deadlock.
  if (world_rank != 0) {
    array = alloc_2d_init(r, c);
    MPI_Recv(&(array[0][0]), r*c, MPI_FLOAT, world_rank - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    //print_helper(array, r, c);

    auto time = endTimer();
    std::cout << "time spend between rank [" << world_rank-1 << "] and [" << world_rank << "] is :"<< time << " seconds \n";
    compute(r, c, array, world_rank);
  } else {
    array = alloc_2d_init(r, c);
    data_init(array, r, c);
    std::cout << "array is generated at rank [" << world_rank << "]'s host.\n";
    compute(r, c, array, world_rank);
  }
  
  startTimer(); 
  MPI_Send(&(array[0][0]), r*c, MPI_FLOAT, (world_rank + 1) % world_size, 0,
           MPI_COMM_WORLD);
  // Now process 0 can receive from the last process. This makes sure that at
  // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
  // deadlock)
  if (world_rank == 0) {
    MPI_Recv(&(array[0][0]), r*c, MPI_FLOAT, world_size - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    std::cout << "array is finally received at rank [" << world_rank << "]'s host.\n";
  }
  MPI_Finalize();
}
