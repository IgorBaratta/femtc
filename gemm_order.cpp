
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

template <typename T>
std::string type_name()
{
    if (std::is_same<T, float>::value)
        return "float";
    else if (std::is_same<T, double>::value)
        return "double";
    else
        return "unknown";
}

// create main
int main(int argc, char **argv)
{
    constexpr int NB = 8;
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
        std::string order_str;

        // get number of dofs, polynomial degree, and loop order from command line
        if (argc > 3)
        {
            Ndofs = atoi(argv[1]);
            degree = atoi(argv[2]);
            order_str = argv[3];
            order = linalg::string2order(order_str);
        }
        else
        {
            // if number of dofs is not given, throw error
            if (rank == 0)
                std::cout << "Usage: " << argv[0] << " <Ndofs> <degree> <order>" << std::endl;
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
            throw std::runtime_error("Number of cells is 0");
        if (degree < 1)
            throw std::runtime_error("Degree must be at least 1");

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

        double elapsed = 0.0;
        // get time from MPI_Wtime()
        MPI_Barrier(comm);
        for (int i = 0; i < 10; i++)
        {
            double t0 = MPI_Wtime();
            linalg::batched_gemm<T, NB>(phi, U, W, num_cells, degree, order);
            double t1 = MPI_Wtime();
            MPI_Barrier(comm);
            elapsed += t1 - t0;
        }
        elapsed /= 10.0;

        // check that all values in W are positive and force writing it back to
        // main memory
        for (std::size_t i = 0; i < W.size(); i++)
            if (W[i] < 0)
                throw std::runtime_error("W is negative");

        // Compute FLOPs
        int m = p + 1;
        int n = (p + 1) * (p + 1);
        int k = p + 1;

        double flops = 3 * (2.0 * m * n * k) * num_cells;
        // Compute memory access
        double mem_access = (U.size() + 2 * W.size()) * sizeof(T);

        double GFLOPs = flops / (elapsed) / 1e9;
        double GBs = mem_access / (elapsed) / 1e9;
        double Gdofs = (num_cells * ndofs) / (elapsed) / 1e9;

        // Sum over all ranks
        double GFLOPs_sum = 0.0;
        double GBs_sum = 0.0;
        double Gdofs_sum = 0.0;
        MPI_Reduce(&GFLOPs, &GFLOPs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&GBs, &GBs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&Gdofs, &Gdofs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

        bool table = true;
        // print time
        if (rank == 0)
        {
            if (table)
            {
                std::cout << degree << " " << size << " " << type_name<T>() << " ";
                std::cout << Ndofs << " " << elapsed << " " << order_str << " ";
                std::cout << GFLOPs_sum << " " << GBs_sum << " " << Gdofs_sum << " ";
                std::cout << NB << std::endl;
            }
            else
            {
                std::cout << "Degree: " << degree << std::endl;
                std::cout << "Comm size: " << size << std::endl;
                std::cout << "Scalar: " << type_name<T>() << std::endl;
                std::cout << "Ndofs: " << U.size() << std::endl;
                std::cout << "Time: " << elapsed << std::endl;
                std::cout << "Loop Order: " << order_str << std::endl;
                std::cout << "GFLOP/s: " << GFLOPs_sum << std::endl;
                std::cout << "GB/s: " << GBs_sum << std::endl;
                std::cout << "GDOF/s: " << Gdofs_sum << std::endl;
            }
        }
    }

    MPI_Finalize();

    return 0;
}