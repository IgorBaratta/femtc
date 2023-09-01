// test micro kernel against reference kernel

#include <iostream>
#include <vector>
#include <algorithm>

#include "tm.hpp"

using T = double;

// create main
int main(int argc, char **argv)
{

    constexpr int P = 3;
    // A has size (P + 1, P + 1)
    // B has size (P + 1, (P + 1) * (P + 1) )
    // C has size (P + 1, (P + 1) * (P + 1) )

    // A is stored in column-major order
    // B is stored in row-major order
    // C is stored in column-major order

    std::vector<double> A((P + 1) * (P + 1));
    std::generate(A.begin(), A.end(), []()
                  { return std::rand() / (T)RAND_MAX; });

    std::vector<double> B((P + 1) * (P + 1) * (P + 1));
    std::generate(B.begin(), B.end(), []()
                  { return std::rand() / (T)RAND_MAX; });

    std::vector<double> C_ref((P + 1) * (P + 1) * (P + 1));

    int lda = P + 1;
    int ldb = (P + 1) * (P + 1);
    int ldc = (P + 1) * (P + 1);

    constexpr int k = P + 1;
    constexpr int m = P + 1;
    constexpr int n = (P + 1) * (P + 1);

    // C_ref = A * B
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t l = 0; l < k; ++l)
                C_ref[i * ldc + j] += A[i + l * lda] * B[l * ldb + j];

    std::vector<double> C((P + 1) * (P + 1) * (P + 1), 0);
    linalg::gemm_blocked<T, k, m, n>(A.data(), B.data(), C.data());

    // check if C == C_ref
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (std::abs(C[i + j * m] - C_ref[i * n + j]) > 1e-10)
                std::cout << "C[" << i + j * m << "] = " << C[i + j * m] << " != " << C_ref[i * n + j] << std::endl;

    return 0;
}