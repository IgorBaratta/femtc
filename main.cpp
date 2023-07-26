// create main.cpp
#include <iostream>
#include <vector>
#include <array>
#include <chrono>

#include "tm.hpp"
#include "tensor.hpp"

using T = double;

int main()
{

    // read input from terminal
    constexpr int p = 12;

    Tensor<T, p + 1, p + 1, p + 1> u;
    Tensor<T, p + 1, p + 1, p + 1> w;
    Matrix<T, p + 1, p + 1> phi;

    phi.fill_random();
    u.fill_random();
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100000; ++i)
            tensor_contraction(phi, u, w);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100000; ++i)
            tensor_contraction_naive(phi, u, w);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time naive: " << elapsed_seconds.count() << "s\n";

        std::cout << phi[0] << std::endl;
    }

    return 0;
}