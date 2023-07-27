// create main.cpp
#include <iostream>
#include <vector>
#include <array>
#include <chrono>

// This block enables to compile the code with and without the likwid header in place
// #ifdef LIKWID_PERFMON
// #include <likwid-marker.h>
// #else
// #define LIKWID_MARKER_INIT
// #define LIKWID_MARKER_THREADINIT
// #define LIKWID_MARKER_REGISTER(regionTag)
// #define LIKWID_MARKER_START(regionTag)
// #define LIKWID_MARKER_STOP(regionTag)
// #define LIKWID_MARKER_CLOSE
// #endif

#include "tm.hpp"
#include "tensor.hpp"

#ifdef DEGREE
constexpr int p = DEGREE;
#else
constexpr int p = 4;
#endif

using T = double;

int main(int argc, char **argv)
{

    // // Register the region "kernel" with LIKWID
    // LIKWID_MARKER_INIT;
    // LIKWID_MARKER_REGISTER("kernel-naive");
    // LIKWID_MARKER_REGISTER("kernel");
    // LIKWID_MARKER_THREADINIT;

    std::vector<T> phi_data((p + 1) * (p + 1), 0);
    {
        Matrix<T, p + 1, p + 1> phi(phi_data.data());

        // Fill the matrices with random values
        phi.fill_random();
        T p0 = phi(0, 0);

        std::cout << p0 << std::endl;

        constexpr int nd = 1e8;
        constexpr int nd_cell = (p + 1) * (p + 1) * (p + 1);
        constexpr int num_cells = nd / nd_cell;

        // allocate data for Tensor (P+1)*(P+1)*(P+1)
        std::vector<T> u_data(num_cells * (p + 1) * (p + 1) * (p + 1));
        std::vector<T> w_data(num_cells * (p + 1) * (p + 1) * (p + 1));

        // // Compute the tensor contraction
        // {
        //     LIKWID_MARKER_START("kernel-naive");
        //     for (int i = 0; i < num_cells; ++i)
        //     {
        //         T *phi = phi_data.data();
        //         T *u = u_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        //         T *w = w_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        //         tensor_contraction_naive<T, p + 1, p + 1>(phi, u, w);
        //     }
        //     LIKWID_MARKER_STOP("kernel-naive");
        // }

        // {
        //     LIKWID_MARKER_START("kernel");
        //     for (int i = 0; i < num_cells; ++i)
        //     {
        //         T *phi = phi_data.data();
        //         T *u = u_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        //         T *w = w_data.data() + i * (p + 1) * (p + 1) * (p + 1);
        //         tensor_contraction<T, p + 1, p + 1>(phi, u, w);
        //     }
        //     LIKWID_MARKER_STOP("kernel");
        // }

        // LIKWID_MARKER_CLOSE;
    }
    return 0;
}