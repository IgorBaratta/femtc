// /// Create a benchmark for the permute function using the Google Benchmark
// /// library.

#include <benchmark/benchmark.h>
#include "permutations.hpp"

using namespace linalg;

// --------------------------------------------------------------------//
/// Create a benchmark for the permute function using the Google Benchmark
/// library.
/// @param[in] state benchmark state
template <int degree, TensorLayout lin, TensorLayout lout>
static void BM_permute(benchmark::State &state)
{
    using T = double;
    constexpr int N0 = degree + 1;
    constexpr int N1 = degree + 1;
    constexpr int N2 = degree + 1;

    std::size_t ncells = state.range(0);
    std::vector<T> A(ncells * N0 * N1 * N2);
    std::vector<T> B(ncells * N0 * N1 * N2);

    // Run the benchmark
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < ncells; i++)
        {
            linalg::permute<T, N0, N1, N1, lin, lout>(A.data(), B.data());
        }
    }
    int num_bytes = sizeof(T);
    int size = ncells * N0 * N1 * N2;
    state.SetItemsProcessed(2 * num_bytes * size * state.iterations());
}

// Register the benchmark
BENCHMARK_TEMPLATE(BM_permute, 5, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6);
BENCHMARK_TEMPLATE(BM_permute, 5, TensorLayout::ijk, TensorLayout::jki)->Range(1e5, 1e6);

// Run the benchmark
int main(int argc, char **argv)
{
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}