// test micro kernel against reference kernel

#include <iostream>
#include <vector>
#include <algorithm>

#include "linalg.hpp"

using T = double;

// create main
int main(int argc, char **argv)
{
    // // A has size (4, k)
    // // B has size (k, 16)
    // // C has size (4, 16)

    // // A is stored in column-major order
    // // B is stored in row-major order
    // // C is stored in row-major order

    // constexpr int k = 26;
    // std::vector<double> A(4 * k);
    // std::generate(A.begin(), A.end(), []()
    //               { return std::rand() / (T)RAND_MAX; });

    // std::vector<double> B(k * 16);
    // std::generate(B.begin(), B.end(), []()
    //               { return std::rand() / (T)RAND_MAX; });

    // std::vector<double> C_ref(4 * 16);

    // // C_ref = A * B
    // for (std::size_t i = 0; i < 4; ++i)
    //     for (std::size_t j = 0; j < 16; ++j)
    //         for (std::size_t l = 0; l < k; ++l)
    //             C_ref[i * 16 + j] += A[i + l * 4] * B[l * 16 + j];

    // std::vector<double> C(4 * 16, 0.0);
    // int lda = 4;
    // int ldb = 16;
    // int ldc = 16;
    // linalg::impl::micro_kernel<T, k>(A.data(), B.data(), C.data(), lda, ldb, ldc);

    // // check if C == C_ref
    // for (std::size_t i = 0; i < 4 * 16; ++i)
    //     if (std::abs(C[i] - C_ref[i]) > 1e-12)
    //         std::cout << "C[" << i << "] = " << C[i] << " != " << C_ref[i] << std::endl;

    // // set C to zero
    // std::fill(C.begin(), C.end(), 0.0);
    // linalg::micro_gemm<T, k, 4, 16>(A.data(), B.data(), C.data(), lda, ldb, ldc);

    // // check if C == C_ref
    // for (std::size_t i = 0; i < 4 * 16; ++i)
    //     if (std::abs(C[i] - C_ref[i]) > 1e-12)
    //         std::cout << "C[" << i << "] = " << C[i] << " != " << C_ref[i] << std::endl;

    // return 0;
}