#include <iostream>
#include "tensor.hpp"

#define restrict __restrict__

// typedef float float4 __attribute__((ext_vector_type(4)));
// typedef float float8 __attribute__((ext_vector_type(8)));
// typedef float float16 __attribute__((ext_vector_type(16)));
// typedef double double4 __attribute__((ext_vector_type(4)));
// typedef double double8 __attribute__((ext_vector_type(8)));

namespace linalg
{
// --------------------------------------------------------------------//
// A is a column major matrix
#define A_(i, j) a[(i) + (j)*m]

// B is a row major matrix
#define B_(i, j) b[(i)*nc + (j)]

// C is a row major matrix
#define C_(i, j) c[(i) + (j)*nc]

    // --------------------------------------------------------------------//
    /// Compute the matrix product C <- A B + C using outer product
    /// @param[in] a matrix of shape (m, k) - column major
    /// @param[in] b matrix of shape (k, n) - row major
    /// @param[out] c matrix of shape (m, n) - column major
    template <typename T, int k, int m, int nc, TensorLayout layout = TensorLayout::ijk>
    void micro_gemm(const T *restrict a, const T *restrict b, T *restrict c)
    {
        if constexpr (layout == TensorLayout::ijk)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < nc; j++)
                    for (int p = 0; p < k; p++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
        else if constexpr (layout == TensorLayout::ikj)
        {
            for (int i = 0; i < m; i++)
                for (int p = 0; p < k; p++)
                    for (int j = 0; j < nc; j++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
        else if constexpr (layout == TensorLayout::jik)
        {
            for (int j = 0; j < nc; j++)
                for (int i = 0; i < m; i++)
                    for (int p = 0; p < k; p++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
        else if constexpr (layout == TensorLayout::jki)
        {
            for (int j = 0; j < nc; j++)
                for (int p = 0; p < k; p++)
                    for (int i = 0; i < m; i++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
        else if constexpr (layout == TensorLayout::kij)
        {
            for (int p = 0; p < k; p++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < nc; j++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
        else if constexpr (layout == TensorLayout::kji)
        {
            for (int p = 0; p < k; p++)
                for (int j = 0; j < nc; j++)
                    for (int i = 0; i < m; i++)
                        C_(i, j) = A_(i, p) * B_(p, j) + C_(i, j);
        }
    }

#undef A_
#undef B_
#undef C_

    // /// Compute a 4x16 block of C using a vectorized dot product.
    // /// @param[in] a matrix of shape (m, k) - column major
    // /// @param[in] b matrix of shape (k, n) - row major
    // /// @param[out] c matrix of shape (m, n) - column major
    // template <int k>
    // void micro_gemm<double, k, 4, 16>(const double *restrict a, const double *restrict b, double *restrict c)
    // {
    //     double8 tmp0[4] = {0.0};
    //     double8 tmp1[4] = {0.0};
    // }

} // namespace

// // --------------------------------------------------------------------//
// /// Compute the tensor contraction C[a,b,c] <- Phi[a, k] U[k, b, c]
// /// @param[in] Phi tensor of shape (Na, Nk)
// /// @param[in] U tensor of shape (Nk, Nb, Nc)
// /// @param[out] C tensor of shape (Na, Nb, Nc)
// template <typename T, std::size_t Nq, std::size_t Nd, TensorLayout layout = default_layout>
// void tensor_contraction_naive(T *restrict phi, T *restrict u, T *restrict w)
// {
//     Matrix<T, Nd, Nq> PHI(phi);
//     Tensor<T, Nd, Nd, Nd, layout> U(u);
//     Tensor<T, Nq, Nd, Nd, layout> W(w);

//     T temp0[Nq * Nd * Nd];
//     T temp1[Nq * Nd * Nd];

//     {
//         Tensor<T, Nq, Nd, Nd> TEMP0(temp0);
//         Tensor<T, Nq, Nd, Nd> TEMP1(temp1);

//         // Temp0[iq0, i1, i2] <- Phi[iq0, i0] U[i0, i1, i2]
//         for (std::size_t iq0 = 0; iq0 < Nq; iq0++)
//             for (std::size_t i1 = 0; i1 < Nd; i1++)
//                 for (std::size_t i2 = 0; i2 < Nd; i2++)
//                     for (std::size_t i0 = 0; i0 < Nd; i0++)
//                         TEMP0(iq0, i1, i2) += PHI(iq0, i0) * U(i0, i1, i2);

//         // Temp1[iq0, iq1, i2] <- Phi[iq1, i1] Temp0[iq1, i1, i2]
//         for (std::size_t iq0 = 0; iq0 < Nq; iq0++)
//             for (std::size_t iq1 = 0; iq1 < Nq; iq1++)
//                 for (std::size_t i2 = 0; i2 < Nd; i2++)
//                     for (std::size_t i1 = 0; i1 < Nd; i1++)
//                         TEMP1(iq0, iq1, i2) += PHI(iq1, i1) * TEMP0(iq0, i1, i2);

//         // W[iq0, iq1, iq2] <- Phi[iq2, i2] Temp1[iq0, iq1, iq2]
//         for (std::size_t iq0 = 0; iq0 < Nq; iq0++)
//             for (std::size_t iq1 = 0; iq1 < Nq; iq1++)
//                 for (std::size_t iq2 = 0; iq2 < Nq; iq2++)
//                     for (std::size_t i2 = 0; i2 < Nd; i2++)
//                         W(iq0, iq1, iq2) += PHI(iq2, i2) * TEMP1(iq0, iq1, i2);
//     }
// }

// // --------------------------------------------------------------------//
// /// Compute the tensor contraction C[a,b,c] <- Phi[a, k] U[k, b, c]
// /// @param[in] Phi tensor of shape (Na, Nk)
// /// @param[in] U tensor of shape (Nk, Nb, Nc)
// /// @param[out] C tensor of shape (Na, Nb, Nc)
// template <typename T, std::size_t Nq, std::size_t Nd, TensorLayout layout = default_layout>
// void tensor_contraction(T *restrict phi, T *restrict u, T *restrict w)
// {
//     Tensor<T, Nd, Nd, Nd, layout> U(u);
//     Tensor<T, Nq, Nd, Nd, layout> W(w);

//     {
//         // ------------------------------------------------------------------
//         // Contraction index: i0
//         // Temp0[iq0, i1, i2] <- Phi[iq0, i0] U[i0, i1, i2]
//         // Can be interpreted as a matrix multiplication
//         // Temp0[iq0, i1_i2] <- Phi[iq0, i0] U[i0, i1_i2]
//         // where i1_i2 is the index (i1, i2) flattened

//         T temp0[Nq * (Nd * Nd)] = {0.0};    // Temporary storage for the matrix multiplication
//         Tensor<T, Nq, Nd, Nd> TEMP0(temp0); // Wrap the temporary storage in a tensor
//         micro_gemm<T, Nd, Nq, Nd * Nd>(phi, u, temp0);

//         // // ------------------------------------------------------------------
//         // // Contraction index: i1
//         // // Temp1[iq0, iq1, i2] <- Phi[iq1, i1] Temp0[iq0, i1, i2]
//         // // Temp0_T[i1, iq1, i2] <- Temp0[iq0, i1, i2]
//         // // (jik) <- (ijk)
//         // // Temp1[iq0, iq1, i2] < -Phi[iq1, i1] Temp0_T[i1, iq0, i2]
//         T temp0_trans[Nq * Nd * Nd];
//         Tensor<T, Nq, Nd, Nd, TensorLayout::kij> TEMP0_t(temp0_trans);
//         TEMP0.template to_layout<TensorLayout::kij>(TEMP0_t); // transpose

//         // // Temp1[iq1, iq0_i2] <- Phi[iq1, i1] Temp0_T[i1, iq0_i2]
//         // // where iq0_i2 is the index (iq0, i2) flattened
//         // // Final result is stored in Temp1[iq1, iq0, i2] (jik)
//         // T temp1[Nq * Nq * Nd];
//         // Tensor<T, Nq, Nq, Nd, TensorLayout::jik> TEMP1(temp1);

//         // // Can be interpreted as a matrix multiplication
//         micro_gemm<T, Nd, Nq, Nq * Nd>(phi, temp0_trans, w);

//         // // ------------------------------------------------------------------
//         // // Contraction index: i2
//         // // W[iq0, iq1, iq2] <- Phi[iq2, i2] Temp1[iq1, iq0, i2]
//         // // Temp1_T[i2, iq1, iq0] <- Temp1[iq1, iq0, i2]
//         // // (kij) <- (jik)
//         // T temp1_trans[Nq * Nq * Nd];
//         // Tensor<T, Nq, Nq, Nd, TensorLayout::kij> TEM1_t(temp1_trans);
//         // TEMP1.template to_layout<TensorLayout::kij>(TEM1_t); // transpose

//         // // W[iq0, iq1, iq2] <- Phi[iq2, i2] Temp1_T[i2, iq1, iq0]
//         // micro_gemm<T, Nd, Nq, Nq * Nd>(phi, TEM1_t.data_ptr(), w);

//         // W[iq2, iq1, iq0] <- Phi[iq2, i2] Temp1_T[i2, iq1, iq0]
//     }
// }