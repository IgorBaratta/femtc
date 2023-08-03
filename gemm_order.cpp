
#include <random>
#include <algorithm>
#include <array>
#include <vector>
#include <functional>
#include <mpi.h>
#include <iostream>
#include <cassert>

#include "tm.hpp"

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
        int Ndofs = 0;
        int degree = 0;
        linalg::Order order;

        // get number of dofs, polynomial degree, and loop order from command line
        if (argc > 3)
        {
            Ndofs = atoi(argv[1]);
            degree = atoi(argv[2]);
            std::string order_str = argv[3];
            order = linalg::string2order(order_str);
        }
        else
        {
            // if number of dofs is not given, throw error
            if (rank == 0)
            {
                std::cout << "Usage: " << argv[0] << " <Ndofs> <degree> <order>" << std::endl;
            }
            MPI_Finalize();
            return 1;
        }

        // polynomial degree
        int p = degree;

        int ndofs = (p + 1) * (p + 1) * (p + 1);

        // get number of cells
        int num_cells = Ndofs / ndofs;

        // if number of cells = 0, throw error
        if (num_cells == 0)
        {
            throw std::runtime_error("Number of cells is 0");
        }
        if (degree < 1)
        {
            throw std::runtime_error("Degree must be at least 1");
        }

        // shape of U for gemm [N0 x [N1 x N2]] = [k x n]
        // shape of phi for gemm [N0 x N0] -> [m x k]
        // shape of W for gemm [N0 x [N1 x N2]] = [m x n]

        // allocate data for phi and set to random divided by RAND_MAX
        std::vector<T> phi((p + 1) * (p + 1), 0.0);
        std::generate(phi.begin(), phi.end(), []()
                      { return std::rand() / (T)RAND_MAX; });

        // allocate data for U and set to random divided by RAND_MAX
        std::vector<T> U(num_cells * ndofs);
        std::generate(U.begin(), U.end(), []()
                      { return std::rand() / (T)RAND_MAX; });

        // allocate data for W and set to zero
        std::vector<T> W(num_cells * ndofs, 0.0);

        // get time from MPI_Wtime()
        double t0 = MPI_Wtime();
        linalg::batched_gemm<T>(phi, U, W, num_cells, degree, order);
        double t1 = MPI_Wtime();

        // check correctness
        std::vector<T> W_ref(ndofs, 0.0);
        for (int i0 = 0; i0 < p + 1; i0++)
            for (int i1 = 0; i1 < p + 1; i1++)
                for (int i2 = 0; i2 < p + 1; i2++)
                    for (int iq = 0; iq < p + 1; iq++)
                    {
                        W_ref[iq * (p + 1) * (p + 1) + i1 * (p + 1) + i2] += phi[i0 * (p + 1) + iq] * U[i0 * (p + 1) * (p + 1) + i1 * (p + 1) + i2];
                    }

        // print W and W_ref
        if (rank == 0)
        {
            for (int i = 0; i < ndofs; i++)
            {
                if (std::abs(W[i] - W_ref[i]) > 1e-12)
                {
                    throw std::runtime_error("Error in contraction");
                }
            }
        }

        // Compute FLOPs
        int m = p + 1;
        int n = (p + 1) * (p + 1);
        int k = p + 1;
        double flops = 2.0 * m * n * k * num_cells;
        // Compute memory access
        double mem_access = (U.size() + W.size()) * sizeof(T);

        // print time
        if (rank == 0)
        {
            std::cout << "Degree: " << degree << std::endl;
            std::cout << "Ndofs: " << U.size() << std::endl;
            std::cout << "Time: " << t1 - t0 << std::endl;
            std::cout << "GFLOP/s: " << flops / (t1 - t0) / 1e9 << std::endl;
            std::cout << "GB/s: " << mem_access / (t1 - t0) / 1e9 << std::endl;
        }
    }

    return 0;
}