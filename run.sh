#!/bin/bash


# run the benchmark for degrees 1 to 15
# ./build/gemm_order size degree order

# 1M degrees of freedom 

size=1000000
num_procs=19

# loop over degrees
for degree in {1..15}
do
    for order in ijk ikj jik jki kij kji
    do
    # repeat 5 times
        for i in {1..5}
        do
            mpirun -n $num_procs ./build/gemm_order $size $degree $order
        done
    done
done

# run libxsmm benchmark
for degree in {1..15}
do
    for i in {1..5}
    do
        mpirun -n $num_procs ./build/gemm_libxsmm $size $degree
    done
done