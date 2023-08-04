#!/bin/bash


# run the benchmark for degrees 1 to 15
# ./build/gemm_order size degree order

# 1M degrees of freedom 
size=1000000

# loop over degrees
for degree in {1..15}
do
    for order in ijk ikj jik jki kij kji
    do
        ./build/gemm_order $size $degree $order
    done
done

# run libxsmm benchmark
for degree in {1..15}
do
    ./build/gemm_libxsmm $size $degree
done