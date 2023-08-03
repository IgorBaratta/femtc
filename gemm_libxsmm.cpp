
#include <random>
#include <algorithm>
#include <array>
#include <vector>
#include <functional>
#include <libxsmm_source.h>
#include <mpi.h>
#include <iostream>

using T = double;

// create main
int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Get the rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    {
        // get number of dofs and polynomial degree from command line
        int Ndofs = 1e8;
        int degree = 3;

        // if number of dofs is given, set Ndofs to that
        if (argc > 2)
        {
            Ndofs = atoi(argv[1]);
            degree = atoi(argv[2]);
        }
        else
        {
            // if number of dofs is not given, throw error
            if (rank == 0)
            {
                std::cout << "Usage: " << argv[0] << " <Ndofs> <degree>" << std::endl;
            }
            MPI_Finalize();
            return 1;
        }

        // polynomial degree
        int p = degree;

        int N0 = p + 1;
        int N1 = p + 1;
        int N2 = p + 1;

        // Number of degrees of freedom per cell
        int ndofs = N0 * N1 * N2;

        // get number of cells
        int num_cells = Ndofs / ndofs;

        // if number of cells = 0, throw error
        if (num_cells == 0)
        {
            throw std::runtime_error("Number of cells is 0");
        }

        // shape of U for gemm [N0 x [N1 x N2]] = [k x n]
        // shape of phi for gemm [N0 x N0] -> [m x k]
        // shape of W for gemm [N0 x [N1 x N2]] = [m x n]

        // allocate data for phi and set to random divided by RAND_MAX
        std::vector<T> phi(N0 * N0, 0.0);
        std::generate(phi.begin(), phi.end(), []()
                      { return std::rand() / (T)RAND_MAX; });

        // allocate data for U and set to random divided by RAND_MAX
        std::vector<T> U(num_cells * ndofs);
        std::generate(U.begin(), U.end(), []()
                      { return std::rand() / (T)RAND_MAX; });

        // allocate data for W and set to zero
        std::vector<T> W(num_cells * ndofs, 0.0);

        // create libxsmm kernel
        libxsmm_gemm_flags flags = LIBXSMM_GEMM_FLAG_NONE; // LIBXSMM_GEMM_FLAGS('N', 'N');

        // Dimensions of the matrices
        int m = N0;
        int n = N1 * N2;
        int k = N0;
        typedef libxsmm_mmfunction<T> kernel_type;
        kernel_type kernel(flags, m, n, k, 1.0 /*alpha*/, 1.0 /*beta*/);
        assert(kernel);

        // get time from MPI_Wtime()
        double t0 = MPI_Wtime();
        // loop over cells
        for (int cell = 0; cell < num_cells; cell++)
        {
            // get pointer to U for cell i
            const T *U_cell = &U[cell * ndofs];

            // get pointer to W for cell i
            T *W_cell = &W[cell * ndofs];

            // call libxsmm kernel
            kernel(phi.data(), U_cell, W_cell);
        }
        double t1 = MPI_Wtime();

        // Compute FLOPs
        double flops = 2.0 * m * n * k * num_cells;
        // Compute memory access
        double mem_access = (U.size() + W.size()) * sizeof(T);

        // print time
        if (rank == 0)
        {
            std::cout << "Degree: " << degree << std::endl;
            std::cout << "Ndofs: " << Ndofs << std::endl;
            std::cout << "Time: " << t1 - t0 << std::endl;
            std::cout << "GFLOP/s: " << flops / (t1 - t0) / 1e9 << std::endl;
            std::cout << "GB/s: " << mem_access / (t1 - t0) / 1e9 << std::endl;
        }
    }

    return 0;
}