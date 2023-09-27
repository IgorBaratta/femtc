// test micro kernel against reference kernel

#include <algorithm>
#include <iostream>
#include <vector>

#include "linalg.hpp"

using T = double;

// create main
int main(int /*argc*/, char** /*argv*/)
{
  // A has size (4, k)
  // B has size (k, 16)
  // C has size (4, 16)

  // A is stored in column-major order
  // B is stored in row-major order
  // C is stored in row-major order

  constexpr int k = 26;
  constexpr int m = 4;
  constexpr int n = 16;
  constexpr int lda = 4;
  constexpr int ldb = 16;
  constexpr int ldc = 16;

  std::vector<double> A(4 * k);
  std::generate(A.begin(), A.end(), []() { return std::rand() / (T)RAND_MAX; });

  std::vector<double> B(k * 16);
  std::generate(B.begin(), B.end(), []() { return std::rand() / (T)RAND_MAX; });

  std::vector<double> C_ref(4 * 16);

  // C_ref = A * B
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 16; ++j)
      for (std::size_t l = 0; l < k; ++l)
        C_ref[i * ldc + j] += A[i + l * lda] * B[l * ldb + j];

  std::vector<double> C(4 * 16, 0.0);
  linalg::gemm_micro<T, k, m, n, lda, ldb, ldc>(A.data(), B.data(), C.data());

  // check if C == C_ref
  for (std::size_t i = 0; i < 4 * 16; ++i)
    if (std::abs(C[i] - C_ref[i]) > 1e-12)
      std::cout << "C[" << i << "] = " << C[i] << " != " << C_ref[i]
                << std::endl;

  return 0;
}
