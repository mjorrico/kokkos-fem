# Kokkos Playground: The Computation of The Element Matrices of a 3D Finite Element Discretization of the Laplacian on a Mesh of Tetrahedral Elements

1. Make sure `Makefile` points to the correct Kokkos directory
2. To target CPU, build with `make -j4 KOKKOS_DEVICES=OpenMP`
3. To target GPU, build with `make -j4 KOKKOS_DEVICES=Cuda`