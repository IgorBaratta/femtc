#include <iostream>
#include "tensor.hpp"

#define restrict __restrict__

// check if compiler is clang
#if defined(__clang__)
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef double double8 __attribute__((ext_vector_type(8)));
#elif defined(__GNUC__) || defined(__GNUG__)
typedef float float4 __attribute__((vector_size(4 * sizeof(float))));
typedef float float8 __attribute__((vector_size(8 * sizeof(float))));
typedef float float16 __attribute__((vector_size(16 * sizeof(float))));
typedef double double4 __attribute__((vector_size(4 * sizeof(double))));
typedef double double8 __attribute__((vector_size(8 * sizeof(double))));
#endif

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
#define MB 4
#define NB 4
#define KB 4
    // Compute the matrix product with block matrix vector producs
    template <typename T, int k, int m, int n, Order layout = Order::ijk>
    void micro_gemm_b(const T *restrict a, const T *restrict b, T *restrict c)
    {
        constexpr int Nk = k / KB; // number of blocks in k direction
        constexpr int Nm = m / MB; // number of blocks in m direction
        constexpr int Nn = n / NB; // number of blocks in n direction

        constexpr int ldA = k;
        constexpr int ldB = n;
        constexpr int ldC = n;

        for (int i = 0; i < Nm; i++)
        {
            for (int j = 0; j < Nn; j++)
            {
                // Compute block C(i,j) = A(i,p) * B(p,j)
                T Cij[MB * NB] = {0.0};
                for (int p = 0; p < Nk; p++)
                {
                    const T *Aip = a + i * MB * ldA + p * KB;
                    const T *Bpj = b + p * KB * ldB + j * NB;
                    micro_gemm<T, KB, MB, NB, layout>(Aip, Bpj, Cij, ldA, ldB, MB);
                }
                // Copy block C(i,j) to C
                T *Cijp = c + i * MB * ldC + j * NB;
                for (int ii = 0; ii < MB * NB; ii++)
                    Cijp[ii] += Cij[ii];
            }
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
            micro_gemm_b<T, k, m, n, layout>(_phi, U_cell, temp0);
            micro_gemm_b<T, k, m, n, layout>(_phi, temp0, temp1);
            micro_gemm_b<T, k, m, n, layout>(_phi, temp1, W_cell);
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
