#include <benchmark/benchmark.h>
#include <random>
#include <functional>
#include "libxsmm_source.h"

#include "tm.hpp"

// --------------------------------------------------------------------//
/// Create a benchmark for batched gemm using the Google Benchmark library.
/// Computes C = A * B for a batch of matrices. Where A is of size N0 x N0,
/// B is of size N0 x (N1 x N2), and C is of size N0 x (N1 x N2).
/// @param[in] state benchmark state
template <typename T, int degree>
void gemm_libxsmm(benchmark::State &state)
{
    constexpr int N0 = degree + 1;
    constexpr int N1 = degree + 1;
    constexpr int N2 = degree + 1;

    // Number of degrees of freedom per cell
    constexpr int ndofs = N0 * N1 * N2;
    // N is the global number of degrees of freedom
    std::size_t N = state.range(0);
    // Compute the number of cells
    std::size_t ncells = N / ndofs;

    // Allocate memory for the basis function
    std::array<T, N0 * N0> A;

    // Allocate memory for the input and output arrays
    std::vector<T> B(N);
    std::vector<T> C(N, 0.0);

    // Fill the input array with random data using c++ random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::generate(A.begin(), A.end(), std::bind(dis, gen));
    std::generate(B.begin(), B.end(), std::bind(dis, gen));

    // Dimensions of the matrices
    int m = N0;
    int n = N1 * N2;
    int k = N0;

    typedef libxsmm_mmfunction<T> kernel_type;
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m, n, k, 1.0 /*alpha*/, 1.0 /*beta*/);
    assert(kernel);

    // Run the benchmark
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < ncells; i++)
        {
            T *in = B.data() + i * n;
            T *out = C.data() + i * n;
            kernel(A.data(), in, out);
        }
        benchmark::ClobberMemory();
    }
    int num_bytes = sizeof(T);
    int size = ncells * ndofs;

    // Set the number of bytes processed (2 read + 1 write)
    state.SetBytesProcessed(3 * num_bytes * size * state.iterations());
    state.counters["Bytes"] = 2 * num_bytes * size;
}

template <typename T, int degree, linalg::TensorLayout order>
void gemm_loops(benchmark::State &state)
{
    constexpr int N0 = degree + 1;
    constexpr int N1 = degree + 1;
    constexpr int N2 = degree + 1;

    // Number of degrees of freedom per cell
    constexpr int ndofs = N0 * N1 * N2;
    // N is the global number of degrees of freedom
    std::size_t N = state.range(0);
    // Compute the number of cells
    std::size_t ncells = N / ndofs;

    // Allocate memory for the basis function
    std::array<T, N0 * N0> A;

    // Allocate memory for the input and output arrays
    std::vector<T> B(N);
    std::vector<T> C(N, 0.0);

    // Fill the input array with random data using c++ random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::generate(A.begin(), A.end(), std::bind(dis, gen));
    std::generate(B.begin(), B.end(), std::bind(dis, gen));

    // Dimensions of the matrices
    int m = N0;
    int n = N1 * N2;
    int k = N0;

    // Run the benchmark
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < ncells; i++)
        {
            T *in = B.data() + i * n;
            T *out = C.data() + i * n;
            linalg::micro_gemm<T, N0, N1, N2, order>(A.data(), in, out);
        }
        benchmark::ClobberMemory();
    }
    int num_bytes = sizeof(T);
    int size = ncells * ndofs;

    // Set the number of bytes processed (1 read + 1 write)
    state.SetBytesProcessed(2 * num_bytes * size * state.iterations());
    state.counters["Bytes"] = 2 * num_bytes * size;
}

// Register the benchmark Libxsmm
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 1)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 2)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 3)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);

// Register the benchmark Loops
BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);




// Run the benchmark
int main(int argc, char **argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}