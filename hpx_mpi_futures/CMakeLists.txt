find_package(HPX REQUIRED)
find_package(MPI REQUIRED)
add_hpx_executable(hpx_G2_ring 
	SOURCES hpx_G2_ring.cpp
	COMPONENT_DEPENDENCIES iostreams
)
