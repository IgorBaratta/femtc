
#include <benchmark/benchmark.h>
#include <random>
#include <functional>
#include <Fastor/Fastor.h>

#include "permutations.hpp"

using namespace linalg;

enum
{
    I,
    J,
    K
};

// --------------------------------------------------------------------//
/// Create a benchmark for the permute function using the Google Benchmark
/// library.
/// @param[in] state benchmark state
template <typename T, int degree, TensorLayout lin, TensorLayout lout>
static void bench_permute(benchmark::State &state)
{
    constexpr int N0 = degree + 1;
    constexpr int N1 = degree + 1;
    constexpr int N2 = degree + 1;
    constexpr int ndofs = N0 * N1 * N2;

    // Allocate memory for the input and output arrays
    std::size_t Ndofs = state.range(0);

    // compute number of cells
    std::size_t ncells = Ndofs / ndofs;

    std::vector<T> A(ncells * ndofs);
    std::vector<T> B(ncells * ndofs);

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
            T *in = A.data() + i * ndofs;
            T *out = B.data() + i * ndofs;
            linalg::permute<T, N0, N1, N1, lin, lout>(in, out);
        }
        benchmark::ClobberMemory();
    }
    int size = ncells * ndofs;

    state.counters["Ndofs"] = size;
    int nprocs = 1;

    // Set the number of bytes processed (1 read + 1 write)
    double num_Gbytes = 3 * sizeof(T) * size * nprocs / 1e9;
    state.counters["GBytes/s"] = benchmark::Counter(num_Gbytes * state.iterations(), benchmark::Counter::kIsRate);
}

// --------------------------------------------------------------------//
/// Create same benchmark now using Fastor
/// @param[in] state benchmark state
template <typename T, int degree, TensorLayout lin, TensorLayout lout>
void bench_fastor_permute(benchmark::State &state)
{

    constexpr int N0 = degree + 1;
    constexpr int N1 = degree + 1;
    constexpr int N2 = degree + 1;
    constexpr int ndofs = N0 * N1 * N2;

    // Allocate memory for the input and output arrays
    std::size_t Ndofs = state.range(0);

    // compute number of cells
    std::size_t ncells = Ndofs / ndofs;

    std::vector<T> A(ncells * ndofs);
    std::vector<T> B(ncells * ndofs);

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
            Fastor::TensorMap<T, N0, N1, N2> in(A.data() + i * ndofs);
            Fastor::TensorMap<T, N0, N1, N2> out(B.data() + i * ndofs);
            auto b1 = Fastor::permute<Fastor::Index<I, J, K>>(in);
            Fastor::assign(out, b1);
        }
        benchmark::ClobberMemory();
        for (std::size_t i = 0; i < A.size(); i++)
            assert(A[i] == B[i]);
    }

    int size = ncells * ndofs;

    state.counters["Ndofs"] = size;
    int nprocs = 1;
    // Set the number of bytes processed (1 read + 1 write)
    double num_Gbytes = 3 * sizeof(T) * size * nprocs / 1e9;
    state.counters["GBytes/s"] = benchmark::Counter(num_Gbytes * state.iterations(), benchmark::Counter::kIsRate);
}

// Register the benchmark
BENCHMARK_TEMPLATE(bench_permute, double, 1, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 2, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 3, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 4, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 5, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 6, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_permute, double, 7, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(bench_fastor_permute, double, 1, TensorLayout::ijk, TensorLayout::ijk)->Range(1e4, 1e8)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 2, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 3, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 4, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 5, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 6, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bench_fastor_permute, double, 7, TensorLayout::ijk, TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond);

// Run the benchmark
int main(int argc, char **argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}