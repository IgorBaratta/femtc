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
    // A is a column major matrix
    template <typename T, int k, int MB>
    void packA(const T *Matrix, int lda, int id, T A_[k * MB])
    {
        int offset = id * MB;
        for (int j = 0; j < MB; j++)
            for (int i = 0; i < k; i++)
                A_[j * MB + i] = Matrix[j * lda + i + offset];
    }

    // B is a row major matrix
    template <typename T, int k, int NB>
    void packB(const T *Matrix, int ldB, int id, T B_[k * NB])
    {
        int offset = id * NB;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < NB; j++)
                B_[i * NB + j] = Matrix[i * ldB + j + offset];
    }

    // C is a row major matrix
    template <typename T, int MB, int NB>
    void unpackC(T C_[MB * NB], T *Matrix, int ldc, int ir, int jr)
    {
        int row0 = ir * MB;
        int col0 = jr * NB;
        for (int i = 0; i < MB; i++)
            for (int j = 0; j < NB; j++)
                Matrix[(row0 + i) * ldc + j + col0] += C_[i * NB + j];
    }

    // --------------------------------------------------------------------//
    // Compute the matrix product with block matrix matrix products
    // A is a column major matrix
    // B is a row major matrix
    // C is a column major matrix
    template <typename T, int k, int m, int n, Order layout = Order::ijk>
    void gemm_blocked(const T *restrict a, const T *restrict b, T *restrict c)
    {
        constexpr int NB = 16;
        constexpr int MB = 4;

        constexpr int Nm = m / MB; // number of blocks in m direction
        constexpr int Nn = n / NB; // number of blocks in n direction

        [[maybe_unused]] constexpr int nrem = n % NB; // size of the last block in n direction
        [[maybe_unused]] constexpr int mrem = m % MB; // size of the last block in m direction

        constexpr int ldA = m; // (p + 1)
        constexpr int ldB = n; // (p + 1) * (p + 1)
        constexpr int ldC = m; // (p + 1)

        T Ctemp[MB * NB] = {0.0};
        for (int jb = 0; jb < Nn; jb++)
        {
            for (int ib = 0; ib < Nm; ib++)
            {
                // pack A
                const T *Aik = a + ib * MB;

                // pack B
                const T *Bpj = b + jb * NB;

                // Compute Cij += Ai. * B.j
                // Ctemp is a row major matrix of size (MB, NB)
                micro_gemm<T, k, MB, NB, layout>(Aik, Bpj, Ctemp, ldA, ldB, NB);

                // pack C
                T *Cij = c + jb * (ldC * NB) + ib * MB;
                // transpose Ctemp and insert into Cij
                for (int i = 0; i < MB; i++)
                    for (int j = 0; j < NB; j++)
                        Cij[j * ldC + i] += Ctemp[i * NB + j];
            }
        }

        if constexpr (mrem > 0)
        {
            const T *Aik = a + Nm * MB;
            for (int jb = 0; jb < Nn; jb++)
            {
                const T *Bpj = b + jb * NB;
                T *Cij = c + jb * (ldC * NB) + Nm * MB;
                micro_gemm<T, k, mrem, NB, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
            }
        }

        if constexpr (nrem > 0)
        {
            const T *Bpj = b + Nn * NB;
            for (int ib = 0; ib < Nm; ib++)
            {
                const T *Aik = a + ib * MB;
                T *Cij = c + Nn * (ldC * NB) + ib * MB;
                micro_gemm<T, k, MB, nrem, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
            }
        }

        if constexpr (mrem > 0 and nrem > 0)
        {
            const T *Aik = a + Nm * MB;
            const T *Bpj = b + Nn * NB;
            T *Cij = c + Nn * (ldC * NB) + Nm * MB;
            micro_gemm<T, k, mrem, nrem, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
        }
    }

    // --------------------------------------------------------------------//
    template <typename T, int P, Order layout = Order::ijk>
    void mass_operator(const std::vector<T> &phi, std::vector<T> &U, std::vector<T> &W, const std::vector<T> &detJ, int num_cells)
    {
        constexpr int ndofs = (P + 1) * (P + 1) * (P + 1);
        constexpr int m = P + 1;
        constexpr int n = (P + 1) * (P + 1);
        constexpr int k = (P + 1);
        for (int cell = 0; cell < num_cells; cell++)
        {
            const T *U_cell = &U[cell * ndofs];
            const T *_phi = &phi[0];
            T *W_cell = &W[cell * ndofs];
            const T *detJ_cell = &detJ[cell * ndofs];
            T temp0[ndofs] = {0.0};
            T temp1[ndofs] = {0.0};
            T W0[ndofs] = {0.0};

            gemm_blocked<T, k, m, n, layout>(_phi, U_cell, temp0);
            gemm_blocked<T, k, m, n, layout>(_phi, temp0, temp1);
            gemm_blocked<T, k, m, n, layout>(_phi, temp1, W0);

            for (int i = 0; i < ndofs; i++)
                W0[i] += W0[i] * detJ_cell[i];

            std::fill_n(temp0, ndofs, 0.0);
            std::fill_n(temp1, ndofs, 0.0);
            gemm_blocked<T, k, m, n, layout>(_phi, W0, temp0);
            gemm_blocked<T, k, m, n, layout>(_phi, temp0, temp1);
            gemm_blocked<T, k, m, n, layout>(_phi, temp1, W_cell);
        }
    }

    // --------------------------------------------------------------------//
    template <typename T, Order layout = Order::ijk>
    void mass_operator(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, std::vector<T> &detJ, int num_cells, int degree)
    {
        // from 1 to 15
        switch (degree)
        {
        case 1:
            mass_operator<T, 1, layout>(a, b, c, detJ, num_cells);
            break;
        case 2:
            mass_operator<T, 2, layout>(a, b, c, detJ, num_cells);
            break;
        case 3:
            mass_operator<T, 3, layout>(a, b, c, detJ, num_cells);
            break;
        case 4:
            mass_operator<T, 4, layout>(a, b, c, detJ, num_cells);
            break;
        case 5:
            mass_operator<T, 5, layout>(a, b, c, detJ, num_cells);
            break;
        case 6:
            mass_operator<T, 6, layout>(a, b, c, detJ, num_cells);
            break;
        case 7:
            mass_operator<T, 7, layout>(a, b, c, detJ, num_cells);
            break;
        case 8:
            mass_operator<T, 8, layout>(a, b, c, detJ, num_cells);
            break;
        case 9:
            mass_operator<T, 9, layout>(a, b, c, detJ, num_cells);
            break;
        case 10:
            mass_operator<T, 10, layout>(a, b, c, detJ, num_cells);
            break;
        case 11:
            mass_operator<T, 11, layout>(a, b, c, detJ, num_cells);
            break;
        case 12:
            mass_operator<T, 12, layout>(a, b, c, detJ, num_cells);
            break;
        case 13:
            mass_operator<T, 13, layout>(a, b, c, detJ, num_cells);
            break;
        case 14:
            mass_operator<T, 14, layout>(a, b, c, detJ, num_cells);
            break;
        case 15:
            mass_operator<T, 15, layout>(a, b, c, detJ, num_cells);
            break;
        case 16:
            mass_operator<T, 16, layout>(a, b, c, detJ, num_cells);
            break;
        case 17:
            mass_operator<T, 17, layout>(a, b, c, detJ, num_cells);
            break;
        case 18:
            mass_operator<T, 18, layout>(a, b, c, detJ, num_cells);
            break;
        case 19:
            mass_operator<T, 19, layout>(a, b, c, detJ, num_cells);
            break;
        case 20:
            mass_operator<T, 20, layout>(a, b, c, detJ, num_cells);
            break;
        case 21:
            mass_operator<T, 21, layout>(a, b, c, detJ, num_cells);
            break;
        case 22:
            mass_operator<T, 22, layout>(a, b, c, detJ, num_cells);
            break;
        case 23:
            mass_operator<T, 23, layout>(a, b, c, detJ, num_cells);
            break;
        case 24:
            mass_operator<T, 24, layout>(a, b, c, detJ, num_cells);
            break;
        case 25:
            mass_operator<T, 25, layout>(a, b, c, detJ, num_cells);
            break;
        default:
            std::cout << "degree not supported" << std::endl;
            break;
        }
    }

    template <typename T>
    void mass_operator(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, std::vector<T> &detJ, int num_cells, int degree, Order order = Order::ijk)
    {
        switch (order)
        {
        case Order::ijk:
            mass_operator<T, Order::ijk>(a, b, c, detJ, num_cells, degree);
            break;
        case Order::ikj:
            mass_operator<T, Order::ikj>(a, b, c, detJ, num_cells, degree);
            break;
        case Order::jik:
            mass_operator<T, Order::jik>(a, b, c, detJ, num_cells, degree);
            break;
        case Order::jki:
            mass_operator<T, Order::jki>(a, b, c, detJ, num_cells, degree);
            break;
        case Order::kij:
            mass_operator<T, Order::kij>(a, b, c, detJ, num_cells, degree);
            break;
        case Order::kji:
            mass_operator<T, Order::kji>(a, b, c, detJ, num_cells, degree);
            break;
        default:
            std::cout << "order not supported" << std::endl;
            break;
        }
    }

} // namespace
