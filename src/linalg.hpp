#include <iostream>
#include "tensor.hpp"
#include <memory.h>
#include <cstring>
#include <experimental/simd>

namespace stdx = std::experimental;

#define restrict __restrict__

// A is a column major matrix
#define A_(i, j) a[(i) + (j) * lda]

// B is a row major matrix
#define B_(i, j) b[(i) * ldb + (j)]

// C is a row major matrix
#define C_(i, j) c[(i) * ldc + (j)]

namespace linalg::impl
{
    template <typename T, int k>
    void micro_kernel(const T *restrict a, const T *restrict b, T *restrict c, int lda, int ldb, int ldc)
    {
        stdx::fixed_size_simd<T, 8> tmp0[4] = {T(0.0)};
        stdx::fixed_size_simd<T, 8> tmp1[4] = {T(0.0)};

        // C <= [4, 16] read
        tmp0[0].copy_to(&C_(0, 0), stdx::element_aligned);
        tmp0[1].copy_to(&C_(1, 0), stdx::element_aligned);
        tmp0[2].copy_to(&C_(2, 0), stdx::element_aligned);
        tmp0[3].copy_to(&C_(3, 0), stdx::element_aligned);
        tmp1[0].copy_to(&C_(0, 8), stdx::element_aligned);
        tmp1[1].copy_to(&C_(1, 8), stdx::element_aligned);
        tmp1[2].copy_to(&C_(2, 8), stdx::element_aligned);
        tmp1[3].copy_to(&C_(3, 8), stdx::element_aligned);

        for (int p = 0; p < k; p++)
        {
            // a <= [4, k] read column and broadcast
            stdx::fixed_size_simd<T, 8> a0p = A_(0, p); // = {A_(0, p), A_(1, p), A_(2, p), A_(3, p)};
            stdx::fixed_size_simd<T, 8> a1p = A_(1, p);
            stdx::fixed_size_simd<T, 8> a2p = A_(2, p);
            stdx::fixed_size_simd<T, 8> a3p = A_(3, p);

            stdx::fixed_size_simd<T, 8> b0, b1;
            b0.copy_from(&B_(p, 0), stdx::element_aligned);
            b1.copy_from(&B_(p, 8), stdx::element_aligned);

            tmp0[0] += a0p * b0;
            tmp0[1] += a1p * b0;
            tmp0[2] += a2p * b0;
            tmp0[3] += a3p * b0;
            tmp1[0] += a0p * b1;
            tmp1[1] += a1p * b1;
            tmp1[2] += a2p * b1;
            tmp1[3] += a3p * b1;
        }

        // C <= [4, 16] write
        tmp0[0].copy_to(&C_(0, 0), stdx::element_aligned);
        tmp0[1].copy_to(&C_(1, 0), stdx::element_aligned);
        tmp0[2].copy_to(&C_(2, 0), stdx::element_aligned);
        tmp0[3].copy_to(&C_(3, 0), stdx::element_aligned);
        tmp1[0].copy_to(&C_(0, 8), stdx::element_aligned);
        tmp1[1].copy_to(&C_(1, 8), stdx::element_aligned);
        tmp1[2].copy_to(&C_(2, 8), stdx::element_aligned);
        tmp1[3].copy_to(&C_(3, 8), stdx::element_aligned);
    }
}

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
        Order order;

        if (order_str == "ikj")
            order = linalg::Order::ikj;
        else if (order_str == "ijk")
            order = linalg::Order::ijk;
        else if (order_str == "jik")
            order = linalg::Order::jik;
        else if (order_str == "jki")
            order = linalg::Order::jki;
        else if (order_str == "kij")
            order = linalg::Order::kij;
        else if (order_str == "kji")
            order = linalg::Order::kji;
        else
            throw std::runtime_error("Invalid loop order");

        return order;
    }
    // --------------------------------------------------------------------//

    // --------------------------------------------------------------------//
    /// Compute the matrix product
    /// @param[in] a matrix of shape (m, k) - column major
    /// @param[in] b matrix of shape (k, n) - row major
    /// @param[out] c matrix of shape (m, n) - column major
    template <typename T, int k, int m, int nc, Order layout = Order::ijk>
    void micro_gemm(const T *restrict a, const T *restrict b, T *restrict c, int lda = m, int ldb = nc, int ldc = nc)
    {
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
    // Compute the matrix product with block matrix matrix products
    // A is a column major matrix
    // B is a row major matrix
    // C is a column major matrix
    template <typename T, int k, int m, int n, Order layout, int MB, int NB>
    void gemm_blocked(const T *restrict a, const T *restrict b, T *restrict c)
    {

        constexpr int block_x = (MB == 0) ? m : MB;
        constexpr int block_y = (NB == 0) ? n : NB;

        constexpr int Nm = m / block_x; // number of blocks in m direction
        constexpr int Nn = n / block_y; // number of blocks in n direction

        [[maybe_unused]] constexpr int mrem = m % block_x; // size of the last block in m direction
        [[maybe_unused]] constexpr int nrem = n % block_y; // size of the last block in n direction

        constexpr int ldA = m; // (p + 1)
        constexpr int ldB = n; // (p + 1) * (p + 1)
        constexpr int ldC = m; // (p + 1)

        for (int jb = 0; jb < Nn; jb++)
        {
            for (int ib = 0; ib < Nm; ib++)
            {
                // pack A (extract MB x k block from A)
                const T *Aik = a + ib * block_x;

                // pack B (extract k x NB block from B)
                const T *Bpj = b + jb * block_y;

                // Compute Cij += Ai. * B.j
                T Ctemp[block_x * block_y] = {0.0};
                micro_gemm<T, k, block_x, block_y, layout>(Aik, Bpj, Ctemp, ldA, ldB, block_y);

                // pack C
                T *Cij = c + jb * (ldC * block_y) + ib * block_x;
                for (int i = 0; i < block_x; i++)
                    for (int j = 0; j < block_y; j++)
                        Cij[j * ldC + i] = Ctemp[i * block_y + j];
            }
        }

        if constexpr (mrem > 0)
        {
            const T *Aik = a + Nm * block_x;
            for (int jb = 0; jb < Nn; jb++)
            {
                const T *Bpj = b + jb * block_y;

                T Ctemp[mrem * block_y] = {0.0};
                micro_gemm<T, k, mrem, block_y, layout>(Aik, Bpj, Ctemp, ldA, ldB, block_y);

                T *Cij = c + jb * (ldC * block_y) + Nm * block_x;

                for (int i = 0; i < mrem; i++)
                    for (int j = 0; j < block_y; j++)
                        Cij[j * ldC + i] = Ctemp[i * block_y + j];
            }
        }

        if constexpr (nrem > 0)
        {
            const T *Bpj = b + Nn * block_y;
            for (int ib = 0; ib < Nm; ib++)
            {
                const T *Aik = a + ib * block_x;

                T Ctemp[block_x * nrem] = {0.0};
                micro_gemm<T, k, block_x, nrem, layout>(Aik, Bpj, Ctemp, ldA, ldB, nrem);

                T *Cij = c + Nn * (ldC * block_y) + ib * block_x;

                for (int i = 0; i < block_x; i++)
                    for (int j = 0; j < nrem; j++)
                        Cij[j * ldC + i] = Ctemp[i * nrem + j];
            }
        }

        if constexpr (mrem > 0 and nrem > 0)
        {
            const T *Aik = a + Nm * block_x;
            const T *Bpj = b + Nn * block_y;

            T Ctemp[mrem * nrem] = {0.0};
            micro_gemm<T, k, mrem, nrem, layout>(Aik, Bpj, Ctemp, ldA, ldB, nrem);

            T *Cij = c + Nn * (ldC * block_y) + Nm * block_x;

            for (int i = 0; i < mrem; i++)
                for (int j = 0; j < nrem; j++)
                    Cij[j * ldC + i] = Ctemp[i * nrem + j];
        }
    }
} // namespace
