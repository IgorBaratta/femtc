
#include <benchmark/benchmark.h>
#include <random>
#include <functional>

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

    constexpr int n = N0 * N1 * N2;

    // Allocate memory for the input and output arrays
    std::size_t N = state.range(0);
    std::vector<T> A(N);
    std::vector<T> B(N);

    // Compute the number of cells
    std::size_t ncells = N / n;

    // Fill the input array with random data using c++ random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::generate(A.begin(), A.end(), std::bind(dis, gen));

    // Run the benchmark
    for (auto _ : state)
    {
        for (std::size_t i = 0; i < ncells; i++)
        {
            T *in = A.data() + i * n;
            T *out = B.data() + i * n;
            linalg::permute<T, N0, N1, N1, lin, lout>(in, out);
        }
        benchmark::ClobberMemory();
    }
    int num_bytes = sizeof(T);
    int size = ncells * n;

    // Set the number of bytes processed (1 read + 1 write)
    state.SetBytesProcessed(2 * num_bytes * size * state.iterations());
    state.SetItemsProcessed(size * state.iterations());
    state.counters["Bytes"] = 2 * num_bytes * size;
}

// Register the benchmark
BENCHMARK_TEMPLATE(BM_permute, 1, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 2, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 3, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 4, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 5, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 6, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_permute, 7, TensorLayout::ijk, TensorLayout::ijk)->Range(1e3, 1e6)->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(BM_permute, 5, TensorLayout::ijk, TensorLayout::ikj)->Range(1e3, 1e6);
// BENCHMARK_TEMPLATE(BM_permute, 5, TensorLayout::jik, TensorLayout::jik)->Range(1e3, 1e6);

// Run the benchmark
int main(int argc, char **argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}