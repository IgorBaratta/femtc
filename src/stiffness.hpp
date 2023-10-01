#pragma once

#include "linalg.hpp"
#include <algorithm>
#include <cstddef>
#include <span>

using namespace linalg;

namespace
{

/// Apply the transformation G to the vector (w0, w1, w2)
template <typename T, int Nq>
void transform(const T* __restrict__ G, T* __restrict__ fw0,
               T* __restrict__ fw1, T* __restrict__ fw2)
{
  for (int iq = 0; iq < Nq; ++iq)
  {
    const T* _G = G + iq * 6;
    const T w0 = fw0[iq];
    const T w1 = fw1[iq];
    const T w2 = fw2[iq];

    fw0[iq] = (_G[5] * w0 + _G[4] * w1 + _G[2] * w2);
    fw1[iq] = (_G[4] * w0 + _G[3] * w1 + _G[1] * w2);
    fw2[iq] = (_G[2] * w0 + _G[1] * w1 + _G[0] * w2);
  }
}
} // namespace

namespace operators
{
template <typename T, int P, Order order, int Mb, int Nb>
void stiffness_operator(std::span<const T> phi, std::span<T> U, std::span<T> W,
                        std::span<const T> G, std::size_t num_cells)
{
  constexpr int ndofs = (P + 1) * (P + 1) * (P + 1);
  constexpr int Nq = (P + 1) * (P + 1) * (P + 1);
  constexpr int m = P + 1;
  constexpr int n = (P + 1) * (P + 1);
  constexpr int k = (P + 1);

  const T* _phi = &phi[0];
  const T* _grad = &phi[(P + 1) * (P + 1)];

  const T* _phi_t = &phi[0];
  const T* _grad_t = &phi[(P + 1) * (P + 1)];

  constexpr int geom_size = 6;

  for (std::size_t cell = 0; cell < num_cells; cell++)
  {

    int stride = cell * ndofs;
    const T* U_cell = &U[stride];
    const T* G_cell = &G[stride * geom_size];
    T* W_cell = &W[stride];

    // allocate temporary arrays
    T temp0[ndofs] = {0.0};
    T temp1[ndofs] = {0.0};

    T Wx[ndofs] = {0.0};
    T Wy[ndofs] = {0.0};
    T Wz[ndofs] = {0.0};

    gemm_blocked<T, k, m, n, order, Mb, Nb>(_grad, U_cell, temp0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, temp1, Wx);

    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, U_cell, temp0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_grad, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, temp1, Wy);

    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi, temp1, Wz);

    transform<T, Nq>(G_cell, Wx, Wy, Wz);

    std::fill_n(temp0, ndofs, 0.0);
    std::fill_n(temp1, ndofs, 0.0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_grad_t, Wx, temp0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, temp1, W_cell);

    std::fill_n(temp0, ndofs, 0.0);
    std::fill_n(temp1, ndofs, 0.0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, Wy, temp0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_grad_t, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, temp1, W_cell);

    std::fill_n(temp0, ndofs, 0.0);
    std::fill_n(temp1, ndofs, 0.0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, Wz, temp0);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_phi_t, temp0, temp1);
    gemm_blocked<T, k, m, n, order, Mb, Nb>(_grad_t, temp1, W_cell);
  }
}

// --------------------------------------------------------------------//
template <typename T, Order order, int Mb, int Nb>
void stiffness_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                        std::span<T> G, int num_cells, int degree)
{
  // from 1 to 25
  switch (degree)
  {
  case 1:
    stiffness_operator<T, 1, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 2:
    stiffness_operator<T, 2, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 3:
    stiffness_operator<T, 3, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 4:
    stiffness_operator<T, 4, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 5:
    stiffness_operator<T, 5, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 6:
    stiffness_operator<T, 6, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 7:
    stiffness_operator<T, 7, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 8:
    stiffness_operator<T, 8, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 9:
    stiffness_operator<T, 9, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 10:
    stiffness_operator<T, 10, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 11:
    stiffness_operator<T, 11, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 12:
    stiffness_operator<T, 12, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 13:
    stiffness_operator<T, 13, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 14:
    stiffness_operator<T, 14, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 15:
    stiffness_operator<T, 15, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 16:
    stiffness_operator<T, 16, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 17:
    stiffness_operator<T, 17, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 18:
    stiffness_operator<T, 18, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 19:
    stiffness_operator<T, 19, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 20:
    stiffness_operator<T, 20, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 21:
    stiffness_operator<T, 21, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 22:
    stiffness_operator<T, 22, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 23:
    stiffness_operator<T, 23, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 24:
    stiffness_operator<T, 24, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  case 25:
    stiffness_operator<T, 25, order, Mb, Nb>(a, b, c, G, num_cells);
    break;
  default:
    std::cout << "degree not supported" << std::endl;
    break;
  }
}

template <typename T, int Mb, int Nb>
void stiffness_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                        std::span<T> G, int num_cells, int degree, Order order)
{
  switch (order)
  {
  case Order::ijk:
    stiffness_operator<T, Order::ijk, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  case Order::ikj:
    stiffness_operator<T, Order::ikj, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  case Order::jik:
    stiffness_operator<T, Order::jik, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  case Order::jki:
    stiffness_operator<T, Order::jki, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  case Order::kij:
    stiffness_operator<T, Order::kij, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  case Order::kji:
    stiffness_operator<T, Order::kji, Mb, Nb>(a, b, c, G, num_cells, degree);
    break;
  default:
    std::cout << "order not supported" << std::endl;
    break;
  }
}

template <typename T, int Nb>
void stiffness_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                        std::span<T> G, int num_cells, int degree, Order order,
                        int Mb)
{
  switch (Mb)
  {
  case 4:
    stiffness_operator<T, 4, Nb>(a, b, c, G, num_cells, degree, order);
    break;
  case 8:
    stiffness_operator<T, 8, Nb>(a, b, c, G, num_cells, degree, order);
    break;
  case 16:
    stiffness_operator<T, 16, Nb>(a, b, c, G, num_cells, degree, order);
    break;
  case 0:
    stiffness_operator<T, 0, Nb>(a, b, c, G, num_cells, degree, order);
    break;
  default:
    std::cout << "Block size not supported" << std::endl;
    break;
  }
}

template <typename T>
void stiffness_operator(std::span<T> a, std::span<T> b, std::span<T> c,
                        std::span<T> G, int num_cells, int degree, Order order,
                        int Mb, int Nb)
{
  switch (Nb)
  {
  case 4:
    stiffness_operator<T, 4>(a, b, c, G, num_cells, degree, order, Mb);
    break;
  case 8:
    stiffness_operator<T, 8>(a, b, c, G, num_cells, degree, order, Mb);
    break;
  case 16:
    stiffness_operator<T, 16>(a, b, c, G, num_cells, degree, order, Mb);
    break;
  case 0:
    stiffness_operator<T, 0>(a, b, c, G, num_cells, degree, order, Mb);
    break;
  default:
    std::cout << "Block size not supported" << std::endl;
    break;
  }
}

} // namespace operators