#!/bin/bash

# get name from command line
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    spack env activate $1

    export LIKWID_HOME=`spack location -i likwid`
    echo $LIKWID_HOME
    export LIBXSMM_HOME=`spack location -i libxsmm`
    echo $LIBXSMM_HOME
    export BENCHMARKS_HOME=`spack location -i benchmark`
    echo $BENCHMARKS_HOME
fi
