// create main.cpp
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

// This block enables to compile the code with and without the likwid
// header in place
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

#include "tensor.hpp"
#include "tm.hpp"

using namespace linalg;

#ifdef DEGREE
constexpr int p = DEGREE;
#else
constexpr int p = 3;
#endif

using T = double;

int main(int argc, char** argv)
{

  // Register the region "kernel" with LIKWID
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_REGISTER("kernel-naive");
  LIKWID_MARKER_REGISTER("kernel");
  LIKWID_MARKER_THREADINIT;

  // allocate data for phi, should probably get this from basix
  std::vector<T> phi_data((p + 1) * (p + 1), 0);
  {
    Matrix<T, p + 1, p + 1> phi(phi_data.data());
    phi.fill_random();

    constexpr std::size_t nd = 1e8;
    constexpr std::size_t nd_cell = (p + 1) * (p + 1) * (p + 1);
    constexpr std::size_t num_cells = nd / nd_cell;

    // allocate data for Tensor (P+1)*(P+1)*(P+1)
    std::vector<T> u_data(num_cells * (p + 1) * (p + 1) * (p + 1));
    std::vector<T> w_data(num_cells * (p + 1) * (p + 1) * (p + 1));

    std::cout << "Using degree " << p << std::endl;
    std::cout << "Size in Gbytes: "
              << (3 * num_cells * (p + 1) * (p + 1) * (p + 1) * sizeof(T)) / 1e9
              << std::endl;

    double timer = 0.0;
    // Compute the tensor contraction
    {
      LIKWID_MARKER_START("kernel-naive");
      auto start = std::chrono::high_resolution_clock::now();
      for (std::size_t i = 0; i < num_cells; ++i)
      {
        T* phi = phi_data.data();
        T* u = u_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        T* w = w_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        tensor_contraction_naive<T, p + 1, p + 1>(phi, u, w);
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto timer = std::chrono::duration<double>(end - start).count();
      std::cout << "Time for naive kernel: " << timer << std::endl;
      LIKWID_MARKER_STOP("kernel-naive");
    }

    {
      LIKWID_MARKER_START("kernel");
      auto start = std::chrono::high_resolution_clock::now();
      for (std::size_t i = 0; i < num_cells; ++i)
      {
        T* phi = phi_data.data();
        T* u = u_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        T* w = w_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        tensor_contraction<T, p + 1, p + 1>(phi, u, w);
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto timer = std::chrono::duration<double>(end - start).count();
      std::cout << "Time for kernel: " << timer << std::endl;
      LIKWID_MARKER_STOP("kernel");
    }

    LIKWID_MARKER_CLOSE;
  }
  return 0;
}
