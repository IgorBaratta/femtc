
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

#include "mass.hpp"

// This block enables to compile the code with and
// without the likwid header in place
#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#endif

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

template <typename T>
void run(int degree, int Ndofs, std::string order_str, int Mb, int Nb)
{

  std::string name = "mass";
  name += typeid(T).name();

  // Register the region "kernel" with LIKWID
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER(name.c_str());
  LIKWID_MARKER_THREADINIT;

  MPI_Comm comm = MPI_COMM_WORLD;

  // Get the rank and size
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  linalg::Order order = linalg::string2order(order_str);

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

  // Allocate data for phi and set to random divided by RAND_MAX should
  // probably get that from
  std::vector<T> phi((p + 1) * (p + 1), 0.0);
  std::generate(phi.begin(), phi.end(),
                []() { return std::rand() / (T)RAND_MAX; });

  // allocate data for U and set to random divided by RAND_MAX
  std::vector<T> U(num_cells * ndofs);
  std::generate(U.begin(), U.end(), []() { return std::rand() / (T)RAND_MAX; });

  // allocate data for W and set to zero
  std::vector<T> W(num_cells * ndofs, 0.0);

  // allocate data for the determinant of the jacobian
  std::vector<T> detJ(num_cells * ndofs, 0.0);
  std::generate(detJ.begin(), detJ.end(),
                []() { return std::rand() / (T)RAND_MAX; });

  double elapsed = 0.0;
  // get time from MPI_Wtime()
  MPI_Barrier(comm);

  LIKWID_MARKER_START(name.c_str());
  for (int i = 0; i < 10; i++)
  {
    double t0 = MPI_Wtime();
    operators::mass_operator<T>(phi, U, W, detJ, num_cells, degree, order, Mb,
                                Nb);
    double t1 = MPI_Wtime();
    MPI_Barrier(comm);

    if (i >= 5)
      elapsed += t1 - t0;
  }
  elapsed /= 5.0;

  LIKWID_MARKER_STOP(name.c_str());
  LIKWID_MARKER_CLOSE;

  // Check that all values in W are positive and force writing it back
  // to main memory
  for (std::size_t i = 0; i < W.size(); i++)
    if (W[i] < 0)
      throw std::runtime_error("W is negative");

  // Compute FLOPs
  int m = p + 1;
  int n = (p + 1) * (p + 1);
  int k = p + 1;

  double flops = 3 * 2 * (2.0 * m * n * k + n * k) * num_cells;

  // Compute memory access (U read, W write + read, detJ read)
  double mem_access = (U.size() + 2 * W.size() + detJ.size()) * sizeof(T);

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

  // Print time
  if (rank == 0)
  {
    std::cout << degree << " " << size << " " << type_name<T>() << " ";
    std::cout << Ndofs << " " << elapsed << " " << order_str << " ";
    std::cout << Mb << " " << Nb << " ";
    std::cout << GFLOPs_sum << " " << GBs_sum << " " << Gdofs_sum << std::endl;
  }
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
    int Ndofs = 0;
    int degree = 0;
    std::string order_str, type_str;
    int Mb = 0;
    int Nb = 0;

    // Get number of dofs, polynomial degree, and loop order from
    // command line
    if (argc > 6)
    {
      Ndofs = atoi(argv[1]);
      degree = atoi(argv[2]);
      order_str = argv[3];
      Mb = atoi(argv[4]);
      Nb = atoi(argv[5]);
      type_str = argv[6];
    }
    else
    {
      // If number of dofs is not given, throw error
      if (rank == 0)
      {
        std::cout << "Usage: " << argv[0]
                  << " <Ndofs> <degree> <order> <Mb> <Nb> <type>" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    // run the code for float and double
    if (type_str == "float")
      run<float>(degree, Ndofs, order_str, Mb, Nb);
    else
      run<double>(degree, Ndofs, order_str, Mb, Nb);
  }

  MPI_Finalize();

  return 0;
}
