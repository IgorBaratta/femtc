#include <benchmark/benchmark.h>
#include <random>
#include <algorithm>
#include <array>
#include <vector>
#include <functional>
#include <libxsmm_source.h>
#include <mpi.h>

#include "tm.hpp"
#include "utils.hpp"

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
    std::vector<T> B(ncells * ndofs, 0.0);
    std::vector<T> C(ncells * ndofs, 0.0);

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
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        for (std::size_t i = 0; i < ncells; i++)
        {
            T *in = B.data() + i * ndofs;
            T *out = C.data() + i * ndofs;
            kernel(A.data(), in, out);
        }
        benchmark::ClobberMemory();
        MPI_Barrier(MPI_COMM_WORLD);
        t = MPI_Wtime() - t;
        double avg_time = reduce_time(t);
        // Set the iteration time in seconds
        state.SetIterationTime(avg_time);
    }

    // total number of degrees of freedom
    int size = ncells * ndofs;

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    state.counters["Ndofs"] = size;

    // Set the number of bytes processed (2 read + 1 write)
    double num_gbytes = 3 * sizeof(T) * size * nprocs / 1e9;
    state.counters["GBytes/s"] = benchmark::Counter(num_gbytes * state.iterations(), benchmark::Counter::kIsRate);

    // Set the number of FLOPs processed (*2 for multiply and add)
    double num_flops = 2 * m * n * k * ncells * nprocs / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(num_flops * state.iterations(), benchmark::Counter::kIsRate);
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
    std::vector<T> B(ncells * ndofs, 0.0);
    std::vector<T> C(ncells * ndofs, 0.0);

    // Fill the input array with random data using c++ random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::generate(A.begin(), A.end(), std::bind(dis, gen));
    std::generate(B.begin(), B.end(), std::bind(dis, gen));

    // Dimensions of the matrices
    constexpr int m = N0;
    constexpr int n = N1 * N2;
    constexpr int k = N0;

    // Run the benchmark
    for (auto _ : state)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();
        for (std::size_t i = 0; i < ncells; i++)
        {
            T *in = B.data() + i * ndofs;
            T *out = C.data() + i * ndofs;
            linalg::micro_gemm<T, k, m, n, order>(A.data(), in, out);
        }
        benchmark::ClobberMemory();
        t = MPI_Wtime() - t;
        double avg_time = reduce_time(t);
        state.SetIterationTime(t);

        std::cout << "Time: " << avg_time << std::endl;
    }
    int size = ncells * ndofs;
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    state.counters["Ndofs"] = size;

    // Set the number of bytes processed (2 read + 1 write)
    double num_gbytes = 3 * sizeof(T) * size * state.iterations() * nprocs / 1e9;
    state.counters["GBytes/s"] = benchmark::Counter(num_gbytes, benchmark::Counter::kIsRate);

    // Set the number of FLOPs processed (*2 for multiply and add)
    double num_flops = 2 * m * n * k * ncells * state.iterations() / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(num_flops * nprocs, benchmark::Counter::kIsRate);
}

// // Register the benchmark Libxsmm
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 1)->Range(1e7, 1e9)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 2)->Range(1e7, 1e9)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 3)->Range(1e7, 1e9)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 4)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 5)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 6)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 7)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 8)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 9)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_libxsmm, double, 10)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

// Register the benchmark Loops (ijk) from degree 1 to 10
BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::ijk)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::ikj)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::jik)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::jki)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::kij)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

BENCHMARK_TEMPLATE(gemm_loops, double, 1, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 2, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 3, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 4, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 5, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 6, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 7, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 8, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 9, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();
BENCHMARK_TEMPLATE(gemm_loops, double, 10, linalg::TensorLayout::kji)->Range(1e5, 1e6)->Unit(benchmark::kMillisecond)->UseManualTime();

// Run the benchmark
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    {
        // Only rank will report results
        NullReporter null_reporter;
        int rank;
        MPI_Comm_rank(comm, &rank);
        benchmark::Initialize(&argc, argv);

        if (rank == 0)
            benchmark::RunSpecifiedBenchmarks();
        else
            benchmark::RunSpecifiedBenchmarks(&null_reporter);

        benchmark::Shutdown();
    }
    MPI_Finalize();
    return 0;
}