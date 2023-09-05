// test micro kernel against reference kernel

#include <iostream>
#include <vector>
#include <algorithm>

#include "linalg.hpp"

using T = double;

template <int P>
void test_gemm()
{
    // A is stored in column-major order
    // B is stored in row-major order
    // C is stored in column-major order

    int lda = P + 1;
    int ldb = (P + 1) * (P + 1);
    int ldc = P + 1;

    constexpr int k = P + 1;
    constexpr int m = P + 1;
    constexpr int n = (P + 1) * (P + 1);

    std::vector<T> A(m * k);
    std::generate(A.begin(), A.end(), []()
                  { return std::rand() / (T)RAND_MAX; });

    std::vector<double> B(k * n);
    std::generate(B.begin(), B.end(), []()
                  { return std::rand() / (T)RAND_MAX; });

    std::vector<double> C_ref(m * n, 0);

    // C_ref = A * B
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                C_ref[i + j * ldc] += A[i + l * lda] * B[l * ldb + j];

    std::vector<double> C(m * n, 0);
    linalg::gemm_blocked<T, k, m, n, linalg::Order::kij, 8, 16>(A.data(), B.data(), C.data());

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (std::abs(C[i + j * m] - C_ref[i + j * m]) > 1e-10)
                throw std::runtime_error("gemm_blocked failed");
}

// create main
int main(int argc, char **argv)
{
    {
        test_gemm<1>();
        test_gemm<2>();
        test_gemm<3>();
        test_gemm<4>();
        test_gemm<5>();
        test_gemm<6>();
        test_gemm<7>();
        test_gemm<8>();
        test_gemm<9>();
        test_gemm<10>();
        test_gemm<11>();
        test_gemm<12>();
        test_gemm<13>();
        test_gemm<14>();
        test_gemm<15>();
        test_gemm<16>();
        test_gemm<17>();
        test_gemm<18>();
        test_gemm<19>();
        test_gemm<20>();
        test_gemm<21>();
        test_gemm<22>();
        test_gemm<23>();
        test_gemm<24>();
        test_gemm<25>();
    }
    return 0;
}