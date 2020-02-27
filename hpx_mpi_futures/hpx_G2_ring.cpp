#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/mpi.hpp>
#include <hpx/lcos/future.hpp>

#include <mpi.h>
int main(int argc, char* argv[])
{
// Init MPI
MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

hpx::mpi::enable_user_polling enable_polling;
hpx::mpi::executor exec(MPI_COMM_WORLD);


if(rank == 0)
{
int token_send=42;
hpx::future<int> f_send =
            hpx::async(exec, MPI_Isend, &token_send, 1, MPI_INT, 1, 0);
f_send.get();
}
if(rank == 1)
{
int token_recv;
hpx::future<int> f_recv =
            hpx::async(exec, MPI_Irecv, &token_recv, 1, MPI_INT, 0, 0);
f_recv.get();
hpx::cout << "rank 1 recieved token " << token_recv << " \n"; 
}


MPI_Barrier(MPI_COMM_WORLD);
MPI_Finalize();
}
