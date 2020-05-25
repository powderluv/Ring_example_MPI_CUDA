# Ring_example_MPI_CUDA

### Two nodes GPUDirect RDMA Striping enabled across two HCAs
```
bash-4.2$ ml cuda
bash-4.2$ ml gcc/8.1.1
bash-4.2$ export CC=mpicc
bash-4.2$ export CXX=mpicxx
bash-4.2$ export PAMI_IBV_DEVICE_NAME=mlx5_0:1,mlx5_3:1
bash-4.2$ export PAMI_IBV_DEVICE_NAME_1=mlx5_3:1,mlx5_0:1
bash-4.2$ export PAMI_ENABLE_STRIPING=1
bash-4.2$ jsrun -r1 -c7 -a 1 -n 2 -g 1 --smpiargs="-gpu" ./cvdlauncher.sh ./gpuDirect
```

### Two GPUs Same Node Same NUMA Domain
```
bash-4.2$ cd build
bash-4.2$ cp ../cvdlauncher.sh .
bash-4.2$ jsrun -r 2 -c 7 -a 1 -n 2 -g 1 --smpiargs="-gpu" ./cvdlauncher.sh ./gpuDirect
```

## Reference:
### Networking on Summit: 
https://www.olcf.ornl.gov/wp-content/uploads/2018/12/summit_workshop_zimmer_network.pdf
