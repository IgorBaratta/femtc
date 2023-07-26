#include <iostream>
#include "tensor.hpp"

#define restrict __restrict__

// // --------------------------------------------------------------------//
// A is column major
#define A_(i, j) a[(i) + (j)*m]

// B is row major
#define B_(i, j) b[(i)*nc + (j)]

// C is row major
#define C_(i, j) c[(i)*nc + (j)]

// --------------------------------------------------------------------//
/// Compute the matrix product C <- A B + C using outer product
/// @param[in] a matrix of shape (m, k) - column major
/// @param[in] b matrix of shape (k, n) - row major
/// @param[out] c matrix of shape (m, n) - column major
template <typename T, int k, int m, int nc>
void micro_gemm(const T *restrict a, const T *restrict b, T *restrict c)
{
    asm("# foo");
    for (int p = 0; p < k; p++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < nc; j++)
                C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
    asm("# bar");
}

// --------------------------------------------------------------------//
/// Compute the tensor contraction C[a,b,c] <- Phi[a, k] U[k, b, c]
/// @param[in] Phi tensor of shape (Na, Nk)
/// @param[in] U tensor of shape (Nk, Nb, Nc)
/// @param[out] C tensor of shape (Na, Nb, Nc)
template <typename Tensor0, typename Tensor1, typename Matrix0>
void tensor_contraction_naive(const Matrix0 &phi, const Tensor0 &U, Tensor1 &W)
{
    constexpr std::size_t Na = Tensor1::template size<0>();
    constexpr std::size_t Nb = Tensor1::template size<1>();
    constexpr std::size_t Nc = Tensor1::template size<2>();
    constexpr std::size_t Nk = Tensor0::template size<0>();

    for (std::size_t a = 0; a < Na; ++a)
    {
        for (std::size_t b = 0; b < Nb; ++b)
        {
            for (std::size_t c = 0; c < Nc; ++c)
            {
                for (std::size_t k = 0; k < Nk; ++k)
                {
                    W(a, b, c) += phi(a, k) * U(k, b, c);
                }
            }
        }
    }
}

// --------------------------------------------------------------------//
/// Compute the tensor contraction C[a,b,c] <- Phi[a, k] U[k, b, c]
/// @param[in] Phi tensor of shape (Na, Nk)
/// @param[in] U tensor of shape (Nk, Nb, Nc)
/// @param[out] C tensor of shape (Na, Nb, Nc)
template <typename Tensor0, typename Tensor1, typename Matrix0>
inline void tensor_contraction(const Matrix0 &phi, const Tensor0 &U, Tensor1 &W)
{
    constexpr std::size_t Na = Tensor1::template size<0>();
    constexpr std::size_t Nb = Tensor1::template size<1>();
    constexpr std::size_t Nc = Tensor1::template size<2>();
    constexpr std::size_t Nk = Tensor0::template size<0>();
    constexpr std::size_t Nd = Nb * Nc;

    for (std::size_t i = 0; i < Na; ++i)
        for (std::size_t k = 0; k < Nk; k++)
            for (std::size_t j = 0; j < Nd; ++j)
                W(i, j) += phi(i, k) * U(k, j);
}