#include <iostream>
#include "tensor.hpp"
#include <memory.h>
#include <cstring>
#include <experimental/simd>

namespace stdx = std::experimental;

#define restrict __restrict__

namespace linalg::impl
{
// A is a column major matrix
#define A_(i, j) a[(i) + (j)*lda]

// B is a row major matrix
#define B_(i, j) b[(i)*ldb + (j)]

// C is a row major matrix
#define C_(i, j) c[(i)*ldc + (j)]

    template <typename T, int k>
    void micro_kernel(const T *restrict a, const T *restrict b, T *restrict c, int lda, int ldb, int ldc)
    {
        constexpr int width = 8;
        typedef stdx::fixed_size_simd<T, width> simd_t;

        constexpr int m = 4;
        constexpr int n = 16;

        constexpr int num_blocks = n / width;

        std::array<simd_t, m * num_blocks> tmp;

        // C <= [m, n] read
        for (int i = 0; i < m; i++)
            for (int j = 0; j < num_blocks; j++)
                tmp[i * num_blocks + j].copy_from(&C_(i, j * width), stdx::element_aligned);

        for (int p = 0; p < k; p++)
        {
            std::array<simd_t, m> a_;
            for (std::size_t i = 0; i < a_.size(); i++)
                a_[i] = A_(i, p);

            std::array<simd_t, num_blocks> b_;
            for (std::size_t j = 0; j < b_.size(); j++)
                b_[j].copy_from(&B_(p, j * width), stdx::element_aligned);

            for (int i = 0; i < m; i++)
                for (int j = 0; j < num_blocks; j++)
                    tmp[i * num_blocks + j] += a_[i] * b_[j];
        }

        // C <= [m, n] write
        for (int i = 0; i < m; i++)
            for (int j = 0; j < num_blocks; j++)
                tmp[i * num_blocks + j].copy_to(&C_(i, j * width), stdx::element_aligned);
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
// A is a column major matrix
#define A_(i, j) a[(i) + (j)*lda]

// B is a row major matrix
#define B_(i, j) b[(i)*ldb + (j)]

// C is a row major matrix
#define C_(i, j) c[(i)*ldc + (j)]

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
    void unpackC(const T C_[MB * NB], T *Matrix, int ldc, int i, int j)
    {
        int row0 = i * MB;
        int col0 = j * NB;
        for (int i = 0; i < MB; i++)
            for (int j = 0; j < NB; j++)
                Matrix[(row0 + i) * ldc + j + col0] += C_[i * NB + j];
    }

    // --------------------------------------------------------------------//
    // Compute the matrix product with block matrix vector producs
    template <typename T, int k, int m, int n, Order layout = Order::ijk>
    void gemm_blocked(const T *restrict a, const T *restrict b, T *restrict c)
    {
        constexpr int NB = 16;
        constexpr int MB = 4;

        constexpr int Nm = m / MB; // number of blocks in m direction
        constexpr int Nn = n / NB; // number of blocks in n direction

        [[maybe_unused]] constexpr int nrem = n % NB; // size of the last block in n direction
        [[maybe_unused]] constexpr int mrem = m % MB; // size of the last block in m direction

        constexpr int ldA = k;
        constexpr int ldB = n;
        constexpr int ldC = n;
        for (int jb = 0; jb < Nn; jb++)
        {
            for (int ib = 0; ib < Nm; ib++)
            {
                const T *Aik = a + ib * MB;
                const T *Bpj = b + jb * NB;
                T *Cij = c + (ib * MB) * ldC + jb * NB;
                impl::micro_kernel<T, k>(Aik, Bpj, Cij, ldA, ldB, ldC);
            }
        }

        if constexpr (mrem > 0)
        {
            const T *Aik = a + Nm * MB;
            for (int jb = 0; jb < Nn; jb++)
            {
                const T *Bpj = b + jb * NB;
                T *Cij = c + (Nm * MB) * ldC + jb * NB;
                micro_gemm<T, k, mrem, NB, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
            }
        }

        if constexpr (nrem > 0)
        {
            const T *Bpj = b + Nn * NB;
            for (int ib = 0; ib < Nm; ib++)
            {
                const T *Aik = a + ib * MB;
                T *Cij = c + (ib * MB) * ldC + Nn * NB;
                micro_gemm<T, k, MB, nrem, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
            }
        }

        if constexpr (mrem > 0 and nrem > 0)
        {
            const T *Aik = a + Nm * MB;
            const T *Bpj = b + Nn * NB;
            T *Cij = c + (Nm * MB) * ldC + Nn * NB;
            micro_gemm<T, k, mrem, nrem, layout>(Aik, Bpj, Cij, ldA, ldB, ldC);
        }
    }

    // --------------------------------------------------------------------//
    template <typename T, int P, Order layout = Order::ijk>
    void batched_template(std::vector<T> &phi, std::vector<T> &U, std::vector<T> &W, int num_cells)
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
            T temp0[ndofs] = {0.0};
            T temp1[ndofs] = {0.0};
            gemm_blocked<T, k, m, n, layout>(_phi, U_cell, temp0);
            gemm_blocked<T, k, m, n, layout>(_phi, temp0, temp1);
            gemm_blocked<T, k, m, n, layout>(_phi, temp1, W_cell);
        }
    }

    template <typename T, Order layout = Order::ijk>
    void batched_gemm(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, int num_cells, int degree)
    {
        // from 1 to 15
        switch (degree)
        {
        case 1:
            batched_template<T, 1, layout>(a, b, c, num_cells);
            break;
        case 2:
            batched_template<T, 2, layout>(a, b, c, num_cells);
            break;
        case 3:
            batched_template<T, 3, layout>(a, b, c, num_cells);
            break;
        case 4:
            batched_template<T, 4, layout>(a, b, c, num_cells);
            break;
        case 5:
            batched_template<T, 5, layout>(a, b, c, num_cells);
            break;
        case 6:
            batched_template<T, 6, layout>(a, b, c, num_cells);
            break;
        case 7:
            batched_template<T, 7, layout>(a, b, c, num_cells);
            break;
        case 8:
            batched_template<T, 8, layout>(a, b, c, num_cells);
            break;
        case 9:
            batched_template<T, 9, layout>(a, b, c, num_cells);
            break;
        case 10:
            batched_template<T, 10, layout>(a, b, c, num_cells);
            break;
        case 11:
            batched_template<T, 11, layout>(a, b, c, num_cells);
            break;
        case 12:
            batched_template<T, 12, layout>(a, b, c, num_cells);
            break;
        case 13:
            batched_template<T, 13, layout>(a, b, c, num_cells);
            break;
        case 14:
            batched_template<T, 14, layout>(a, b, c, num_cells);
            break;
        case 15:
            batched_template<T, 15, layout>(a, b, c, num_cells);
            break;

        default:
            std::cout << "degree not supported" << std::endl;
            break;
        }
    }

    template <typename T>
    void batched_gemm(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, int num_cells, int degree, Order order = Order::ijk)
    {
        switch (order)
        {
        case Order::ijk:
            batched_gemm<T, Order::ijk>(a, b, c, num_cells, degree);
            break;
        case Order::ikj:
            batched_gemm<T, Order::ikj>(a, b, c, num_cells, degree);
            break;
        case Order::jik:
            batched_gemm<T, Order::jik>(a, b, c, num_cells, degree);
            break;
        case Order::jki:
            batched_gemm<T, Order::jki>(a, b, c, num_cells, degree);
            break;
        case Order::kij:
            batched_gemm<T, Order::kij>(a, b, c, num_cells, degree);
            break;
        case Order::kji:
            batched_gemm<T, Order::kji>(a, b, c, num_cells, degree);
            break;
        default:
            std::cout << "order not supported" << std::endl;
            break;
        }
    }

} // namespace
