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

//#include "host_to_device.hpp"

int** alloc_2d_init(int r, int c)
{
	int** A = new int*[r];
	A[0] = new  int[r*c];
	for (int i = 1; i < r; ++i) 
		A[i] = A[i-1] + c;
	return A;
}

void data_init(int** A, int r, int c)
{
	for(int i=0; i<r; i++)
	{
		for(int j=0; j<c; j++)
		{
			A[i][j]=i*c+j;
		}
	}
}

void print_helper(int** A, int r, int c)
{
	
	for(int i=0; i<r; i++)
	{
		for(int j=0; j<c; j++)
		{
			printf("%d, ", A[i][j]);
		}
	printf("\n");
	}
}

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int** array;
  int r = 4;
  int c = 4;
  // Receive from the lower process and send to the higher process. Take care
  // of the special case when you are the first process to prevent deadlock.
  if (world_rank != 0) {
    array = alloc_2d_init(r, c);
    MPI_Recv(&(array[0][0]), r*c, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    print_helper(array, r, c);
    printf("array is received at rank [%d], array[1][1] is %d, printing from rank %d's host\n", world_rank, array[1][1], world_rank);
    //compute(N, array, world_rank);
  } else {
    array = alloc_2d_init(r, c);
    data_init(array, r, c);
    printf("array is generated at rank[%d], printing from rank %d's host\n", world_rank, world_rank);
    //compute(N, array, world_rank);
  }
  MPI_Send(&(array[0][0]), r*c, MPI_INT, (world_rank + 1) % world_size, 0,
           MPI_COMM_WORLD);
  // Now process 0 can receive from the last process. This makes sure that at
  // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
  // deadlock)
  if (world_rank == 0) {
    MPI_Recv(&(array[0][0]), r*c, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("array is finally received at rank [%d], array[1][1] is %d, printing from rank %d's host\n", world_rank, array[1][1], world_rank);
  }
  MPI_Finalize();
}
