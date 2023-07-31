cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "tcfem")
project(${PROJECT_NAME} VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")

include_directories("src")

# find google benchmark
find_package(benchmark REQUIRED)

# find likwid
find_package(likwid REQUIRED)

# find BLAS any flavor
find_package(BLAS REQUIRED)

# find MPI
find_package(MPI REQUIRED)

# find libxsmm
if(NOT DEFINED ENV{LIBXSMM_HOME})
    message(FATAL_ERROR "LIBXSMM_HOME is not defined")
else()
    message(STATUS "LIBXSMM_HOME is $ENV{LIBXSMM_HOME}")
    set(XSMM_INCLUDES ${LIBXSMM_HOME}/include)
endif()

# Create a sources variable with a link to all cpp files to compile
set(SOURCES permute.cpp batch_gemm.cpp)

# Add a different executable for each cpp file in SOURCES
foreach(sourcefile ${SOURCES})
    # I used a simple string replace, to cut off .cpp.
    string(REPLACE ".cpp" "" name ${sourcefile})
    add_executable(${name} ${sourcefile})
    target_include_directories(${name} PUBLIC ${XSMM_INCLUDES})
    target_link_libraries(${name} benchmark::benchmark likwid::likwid ${BLAS_LIBRARIES} MPI::MPI_CXX)
endforeach(sourcefile ${SOURCES})