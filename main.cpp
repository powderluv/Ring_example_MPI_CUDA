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

  char* s_array;
  char* r_array;
  int r = 4;
  int c = 4;
  d2d_alloc(r, c, &s_array);
  d2d_alloc(r, c, &r_array);
  // Receive from the lower process and send to the higher process. Take care
  // of the special case when you are the first process to prevent deadlock.
  if (world_rank != 0) {
    MPI_Recv(r_array, r*c, MPI_CHAR, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    d2d_compute(r, c, r_array);

  } else {
    d2d_memset(r,c, s_array);
    MPI_Send(s_array, r*c, MPI_CHAR, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
  }
  
  MPI_Finalize();
}
