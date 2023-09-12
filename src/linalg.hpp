#include "tensor.hpp"
#include <cstring>
#include <experimental/simd>
#include <iostream>
#include <memory.h>

namespace stdx = std::experimental;

#define restrict __restrict__

namespace linalg
{
// Enum for loop order
enum class Order
{
  ijk,
  ikj,
  jik,
  jki,
  kij,
  kji
};

Order string2order(std::string order_str)
{
  if (order_str == "ikj")
    return linalg::Order::ikj;
  else if (order_str == "ijk")
    return linalg::Order::ijk;
  else if (order_str == "jik")
    return linalg::Order::jik;
  else if (order_str == "jki")
    return linalg::Order::jki;
  else if (order_str == "kij")
    return linalg::Order::kij;
  else if (order_str == "kji")
    return linalg::Order::kji;
  else
    throw std::runtime_error("Invalid loop order");
}
// --------------------------------------------------------------------//

// --------------------------------------------------------------------//
/// Compute the matrix product c += a.b
/// @param[in] a matrix of shape (m, k) - column major
/// @param[in] b matrix of shape (k, n) - row major
/// @param[out] c matrix of shape (m, n) - column major
template <typename T, int k, int m, int nc, Order layout = Order::ijk>
void micro_gemm(const T* restrict a, const T* restrict b, T* restrict c,
                int lda = m, int ldb = nc, int ldc = nc)
{
// #define A_(i, j) a[(i) + (j)*lda]
// #define B_(i, j) b[(i)*ldb + (j)]
#define C_(i, j) c[(i)*ldc + (j)]

  auto A_ = [&](auto i, auto j) { return a[i + j * lda]; };
  auto B_ = [&](auto i, auto j) { return b[i * ldb + j]; };
  // auto C_ = [&](auto i, auto j)
  // { return c[i * ldc + j]; };

  if constexpr (layout == Order::ijk)
  {
    for (int i = 0; i < m; i++)
      for (int j = 0; j < nc; j++)
        for (int p = 0; p < k; p++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
  else if constexpr (layout == Order::ikj)
  {
    for (int i = 0; i < m; i++)
      for (int p = 0; p < k; p++)
        for (int j = 0; j < nc; j++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
  else if constexpr (layout == Order::jik)
  {
    for (int j = 0; j < nc; j++)
      for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
  else if constexpr (layout == Order::jki)
  {
    for (int j = 0; j < nc; j++)
      for (int p = 0; p < k; p++)
        for (int i = 0; i < m; i++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
  else if constexpr (layout == Order::kij)
  {
    for (int p = 0; p < k; p++)
      for (int i = 0; i < m; i++)
        for (int j = 0; j < nc; j++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
  else if constexpr (layout == Order::kji)
  {
    for (int p = 0; p < k; p++)
      for (int j = 0; j < nc; j++)
        for (int i = 0; i < m; i++)
          C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
  }
}

// --------------------------------------------------------------------//
// Compute the matrix product C += AB with block matrix matrix products.
// A is a column major matrix
// B is a row major matrix
// C is a column major matrix
template <typename T, int k, int m, int n, Order layout, int MB, int NB>
void gemm_blocked(const T* restrict a, const T* restrict b, T* restrict c)
{
  constexpr int block_x = (MB == 0) ? m : MB;
  constexpr int block_y = (NB == 0) ? n : NB;

  constexpr int Nm = m / block_x; // number of blocks in m direction
  constexpr int Nn = n / block_y; // number of blocks in n direction

  [[maybe_unused]] constexpr int mrem
      = m % block_x; // size of the last block in m direction
  [[maybe_unused]] constexpr int nrem
      = n % block_y; // size of the last block in n direction

  constexpr int ldA = m; // (p + 1)
  constexpr int ldB = n; // (p + 1) * (p + 1)
  constexpr int ldC = m; // (p + 1)

  for (int jb = 0; jb < Nn; jb++) // Columns of B
  {
    for (int ib = 0; ib < Nm; ib++) // Rows of A
    {
      // Pointer to start of A sub-block (extract MB x k block from A)
      const T* Aik = a + ib * block_x;

      // Pointer to start of B sub-block (extract k x NB block from B)
      const T* Bpj = b + jb * block_y;

      // Compute block of Cij += Ai. * B.j
      T Ctemp[block_x * block_y] = {0.0};
      micro_gemm<T, k, block_x, block_y, layout>(Aik, Bpj, Ctemp, ldA, ldB,
                                                 block_y);

      // Copy Ctemp (row-major, block) to C (column-major)
      T* Cij = c + jb * (ldC * block_y) + ib * block_x;
      for (int i = 0; i < block_x; i++)
        for (int j = 0; j < block_y; j++)
          Cij[j * ldC + i] = Ctemp[i * block_y + j];
    }
  }

  if constexpr (mrem > 0)
  {
    const T* Aik = a + Nm * block_x;
    for (int jb = 0; jb < Nn; jb++)
    {
      const T* Bpj = b + jb * block_y;
      T Ctemp[mrem * block_y] = {0.0};
      micro_gemm<T, k, mrem, block_y, layout>(Aik, Bpj, Ctemp, ldA, ldB,
                                              block_y);
      T* Cij = c + jb * (ldC * block_y) + Nm * block_x;
      for (int i = 0; i < mrem; i++)
        for (int j = 0; j < block_y; j++)
          Cij[j * ldC + i] = Ctemp[i * block_y + j];
    }
  }

  if constexpr (nrem > 0)
  {
    const T* Bpj = b + Nn * block_y;
    for (int ib = 0; ib < Nm; ib++)
    {
      const T* Aik = a + ib * block_x;
      T Ctemp[block_x * nrem] = {0.0};
      micro_gemm<T, k, block_x, nrem, layout>(Aik, Bpj, Ctemp, ldA, ldB, nrem);
      T* Cij = c + Nn * (ldC * block_y) + ib * block_x;
      for (int i = 0; i < block_x; i++)
        for (int j = 0; j < nrem; j++)
          Cij[j * ldC + i] = Ctemp[i * nrem + j];
    }
  }

  if constexpr (mrem > 0 and nrem > 0)
  {
    const T* Aik = a + Nm * block_x;
    const T* Bpj = b + Nn * block_y;

    T Ctemp[mrem * nrem] = {0.0};
    micro_gemm<T, k, mrem, nrem, layout>(Aik, Bpj, Ctemp, ldA, ldB, nrem);

    T* Cij = c + Nn * (ldC * block_y) + Nm * block_x;
    for (int i = 0; i < mrem; i++)
      for (int j = 0; j < nrem; j++)
        Cij[j * ldC + i] = Ctemp[i * nrem + j];
  }
}
} // namespace linalg
