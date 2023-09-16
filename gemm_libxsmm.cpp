
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <libxsmm_source.h>
#include <mpi.h>
#include <random>
#include <vector>

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
int main(int argc, char** argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // Get the rank and size
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  {
    // Get number of dofs and polynomial degree from command line
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
    std::generate(phi.begin(), phi.end(),
                  []() { return std::rand() / (T)RAND_MAX; });

    // allocate data for U and set to random divided by RAND_MAX
    std::vector<T> U(num_cells * ndofs);
    std::generate(U.begin(), U.end(),
                  []() { return std::rand() / (T)RAND_MAX; });

    // allocate data for W and set to zero
    std::vector<T> W(num_cells * ndofs, 0.0);

    // create libxsmm kernel
    libxsmm_gemm_flags flags
        = LIBXSMM_GEMM_FLAG_NONE; // LIBXSMM_GEMM_FLAGS('N', 'N');

    // Dimensions of the matrices
    int m = N0;
    int n = N1 * N2;
    int k = N0;
    typedef libxsmm_mmfunction<T> kernel_type;
    kernel_type kernel(flags, m, n, k, 1.0, 1.0);
    assert(kernel);

    double elapsed = 0.0;
    // get time from MPI_Wtime()
    MPI_Barrier(comm);
    for (int i = 0; i < 10; i++)
    {
      double t0 = MPI_Wtime();
      std::vector<T> temp0(N0 * N1 * N2);
      std::vector<T> temp0_transp(N0 * N1 * N2);
      std::vector<T> temp1(N0 * N1 * N2);
      std::vector<T> temp1_transp(N0 * N1 * N2);
      // loop over cells
      for (int cell = 0; cell < num_cells; cell++)
      {
        std::fill(temp0.begin(), temp0.end(), 0.0);
        std::fill(temp1.begin(), temp1.end(), 0.0);

        // get pointer to U for cell i
        const T* U_cell = &U[cell * ndofs];

        // get pointer to W for cell i
        T* W_cell = &W[cell * ndofs];

        // 1st tensor contraction
        // temp0[q0, i1, i2] = phi[q0, i0] * U_cell[i0, i1, i2]
        kernel(phi.data(), U_cell, temp0.data());

        // transpose temp0
        // temp0_transp[i1, i2, q0] = temp0[q0, i1, i2]
        for (int i0 = 0; i0 < N0; i0++)
          for (int i1 = 0; i1 < N1; i1++)
            for (int i2 = 0; i2 < N2; i2++)
              temp0_transp[i1 * (N2 * N0) + i2 * N0 + i0]
                  = temp0[i0 * (N1 * N2) + i1 * N2 + i2];

        // 2nd tensor contraction
        // temp1[q1, i2, q0] = phi[q1, i1] * temp0_transp[i1, i2, q0]
        kernel(phi.data(), temp0_transp.data(), temp1.data());

        // transpose temp1
        // temp1_transp[i2, q1, q0] = temp1[q1, i2, q0]
        for (int i0 = 0; i0 < N0; i0++)
          for (int i1 = 0; i1 < N1; i1++)
            for (int i2 = 0; i2 < N2; i2++)
              temp1_transp[i2 * (N1 * N0) + i1 * N0 + i0]
                  = temp1[i1 * (N2 * N0) + i2 * N0 + i0];

        // 3rd tensor contraction
        // W_cell[q2, q1, q0] += phi[q2, i2] * temp1_transp[i2, q1, q0]
        kernel(phi.data(), temp1_transp.data(), W_cell);
      }
      double t1 = MPI_Wtime();
      MPI_Barrier(comm);
      elapsed += t1 - t0;
    }
    elapsed /= 10.0;

    // Check that all values in W are positive and force writing it back
    // to main memory
    for (std::size_t i = 0; i < W.size(); i++)
      if (W[i] < 0)
        throw std::runtime_error("W is negative");

    double flops = 3 * (2.0 * m * n * k) * num_cells;
    // Compute memory access
    double mem_access = (U.size() + 2 * W.size()) * sizeof(T);

    double GFLOPs = flops / elapsed / 1e9;
    double GBs = mem_access / elapsed / 1e9;
    double Gdofs = (num_cells * ndofs) / elapsed / 1e9;

    // Sum over all ranks
    double GFLOPs_sum = 0.0;
    double GBs_sum = 0.0;
    double Gdofs_sum = 0.0;
    MPI_Reduce(&GFLOPs, &GFLOPs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&GBs, &GBs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&Gdofs, &Gdofs_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    bool table = true;
    std::string order_str = "libxsmm";
    // print time
    if (rank == 0)
    {
      if (table)
      {
        std::cout << degree << " " << size << " " << type_name<T>() << " ";
        std::cout << Ndofs << " " << elapsed << " " << order_str << " ";
        std::cout << GFLOPs_sum << " " << GBs_sum << " " << Gdofs_sum
                  << std::endl;
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
