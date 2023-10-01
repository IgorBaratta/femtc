#include "linalg.hpp"
#include <algorithm>
#include <cstddef>
#include <span>

using namespace linalg;

namespace operators
{
template <typename T, int P, Order layout = Order::ijk, int Mb, int Nb>
void mass_operator(std::span<const T> phi, std::span<T> U, std::span<T> W,
                   std::span<const T> detJ, std::size_t num_cells)
{
  constexpr int ndofs = (P + 1) * (P + 1) * (P + 1);
  constexpr int m = P + 1;
  constexpr int n = (P + 1) * (P + 1);
  constexpr int k = (P + 1);

  // Get basis evaluation matrix and its transpose
  const T* _phi = phi.data();
  const T* _phi_t = phi.data();

  for (std::size_t cell = 0; cell < num_cells; cell++)
  {
    std::size_t stride = cell * ndofs;
    const T* U_cell = &U[stride];
    const T* detJ_cell = &detJ[stride];

    T* W_cell = &W[stride];

    // Temporary arrays
    T temp0[ndofs] = {0.0};
    T temp1[ndofs] = {0.0};
    T W0[ndofs] = {0.0};

    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi, U_cell, temp0);
    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi, temp0, temp1);
    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi, temp1, W0);

    for (int i = 0; i < ndofs; i++)
      W0[i] = W0[i] * detJ_cell[i];

    std::fill_n(temp0, ndofs, 0.0);
    std::fill_n(temp1, ndofs, 0.0);
    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi_t, W0, temp0);
    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi_t, temp0, temp1);
    gemm_blocked<T, k, m, n, layout, Mb, Nb>(_phi_t, temp1, W_cell);
  }
}

// --------------------------------------------------------------------//
template <typename T, Order layout, int Mb, int Nb>
void mass_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                   std::span<T> detJ, std::size_t num_cells, int degree)
{
  // from 1 to 15
  switch (degree)
  {
  case 1:
    mass_operator<T, 1, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 2:
    mass_operator<T, 2, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 3:
    mass_operator<T, 3, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 4:
    mass_operator<T, 4, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 5:
    mass_operator<T, 5, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 6:
    mass_operator<T, 6, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 7:
    mass_operator<T, 7, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 8:
    mass_operator<T, 8, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 9:
    mass_operator<T, 9, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 10:
    mass_operator<T, 10, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 11:
    mass_operator<T, 11, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 12:
    mass_operator<T, 12, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 13:
    mass_operator<T, 13, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 14:
    mass_operator<T, 14, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 15:
    mass_operator<T, 15, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 16:
    mass_operator<T, 16, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 17:
    mass_operator<T, 17, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 18:
    mass_operator<T, 18, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 19:
    mass_operator<T, 19, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 20:
    mass_operator<T, 20, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 21:
    mass_operator<T, 21, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 22:
    mass_operator<T, 22, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 23:
    mass_operator<T, 23, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 24:
    mass_operator<T, 24, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  case 25:
    mass_operator<T, 25, layout, Mb, Nb>(a, b, c, detJ, num_cells);
    break;
  default:
    std::cout << "degree not supported" << std::endl;
    break;
  }
}

template <typename T, int Mb, int Nb>
void mass_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                   std::span<T> detJ, std::size_t num_cells, int degree,
                   Order order)
{
  switch (order)
  {
  case Order::ijk:
    mass_operator<T, Order::ijk, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  case Order::ikj:
    mass_operator<T, Order::ikj, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  case Order::jik:
    mass_operator<T, Order::jik, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  case Order::jki:
    mass_operator<T, Order::jki, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  case Order::kij:
    mass_operator<T, Order::kij, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  case Order::kji:
    mass_operator<T, Order::kji, Mb, Nb>(a, b, c, detJ, num_cells, degree);
    break;
  default:
    std::cout << "order not supported" << std::endl;
    break;
  }
}

template <typename T, int Nb>
void mass_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                   std::span<T> detJ, std::size_t num_cells, int degree,
                   Order order, int Mb)
{
  switch (Mb)
  {
  case 4:
    mass_operator<T, 4, Nb>(a, b, c, detJ, num_cells, degree, order);
    break;
  case 8:
    mass_operator<T, 8, Nb>(a, b, c, detJ, num_cells, degree, order);
    break;
  case 16:
    mass_operator<T, 16, Nb>(a, b, c, detJ, num_cells, degree, order);
    break;
  case 0:
    mass_operator<T, 0, Nb>(a, b, c, detJ, num_cells, degree, order);
    break;
  default:
    std::cout << "Block size not supported" << std::endl;
    break;
  }
}

template <typename T>
void mass_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                   std::span<T> detJ, std::size_t num_cells, int degree,
                   Order order, int Mb, int Nb)
{
  switch (Nb)
  {
  case 4:
    mass_operator<T, 4>(a, b, c, detJ, num_cells, degree, order, Mb);
    break;
  case 8:
    mass_operator<T, 8>(a, b, c, detJ, num_cells, degree, order, Mb);
    break;
  case 16:
    mass_operator<T, 16>(a, b, c, detJ, num_cells, degree, order, Mb);
    break;
  case 0:
    mass_operator<T, 0>(a, b, c, detJ, num_cells, degree, order, Mb);
    break;
  default:
    std::cout << "Block size not supported" << std::endl;
    break;
  }
}

} // namespace operators
