#include <iostream>
#include <thread>

#include "util_mpi.hpp"
#include "util_cuda.hpp"

#define MOD(x,n) ((x) % (n))

void ringG(int thread_id, int rank, int mpi_size, int niter, int n_elems, float* G2, float* sendbuff_G2)
{
    int left_neighbor = MOD((rank-1 + mpi_size), mpi_size);
    int right_neighbor = MOD((rank+1 + mpi_size), mpi_size);

    MPI_Request recv_request;
    MPI_Request send_request;
    MPI_Status status;

    for(int i = 0; i < niter; i++)
    {
      // generate G2 and fill some value in
      generateG2(G2, rank, n_elems);

       // get ready for send
      CudaMemoryCopy(sendbuff_G2, G2, n_elems);
      int send_tag = thread_id;
      int recv_tag = thread_id;
      for(int icount=0; icount < (mpi_size-1); icount++)
      {
	std::cout << "\nthread " << thread_id <<" from rank " << rank << "\n"; 
        MPI_CHECK(MPI_Irecv(G2, n_elems, MPI_FLOAT, left_neighbor, recv_tag, MPI_COMM_WORLD, &recv_request));
        MPI_CHECK(MPI_Isend(sendbuff_G2, n_elems, MPI_FLOAT, right_neighbor, send_tag, MPI_COMM_WORLD, &send_request));
        MPI_CHECK(MPI_Wait(&recv_request, &status));
        MPI_CHECK(MPI_Wait(&send_request, &status)); // wait for sendbuf_G2 to be available again
        CudaMemoryCopy(sendbuff_G2, G2, n_elems);
        }
    }
  
}

int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, mpi_size;
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int n_elems = 10;
  int n_iter = 1;
  float* G2 = nullptr;
  float* sendbuff_G2 = nullptr;

  G2 = allocate_on_device<float>(n_elems);
  sendbuff_G2 = allocate_on_device<float>(n_elems);

  std::thread first(ringG, 1, rank, mpi_size, n_iter, n_elems, G2, sendbuff_G2);
  std::thread second(ringG, 2, rank, mpi_size, n_iter, n_elems, G2, sendbuff_G2);

  first.join();
  second.join();

  // sync all processors at the end
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
