cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "tcfem")
project(${PROJECT_NAME} VERSION 1.0.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Wall -Wfatal-errors")

include_directories("src")

# clone repository https://github.com/romeric/Fastor
set(FASTOR_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/Fastor)
message(STATUS "Fastor include dir is ${FASTOR_INCLUDE_DIR}")

# find likwid
# find_package(likwid REQUIRED)
find_package(likwid)

# find BLAS any flavor
find_package(BLAS REQUIRED)

# find MPI
find_package(MPI REQUIRED)

# # find libxsmm
# if(NOT DEFINED ENV{LIBXSMM_HOME})
#     message(FATAL_ERROR "LIBXSMM_HOME is not defined")
# else()
#     message(STATUS "LIBXSMM_HOME is $ENV{LIBXSMM_HOME}")
#     set(XSMM_INCLUDES ${LIBXSMM_HOME}/include)
# endif()

# Create a sources variable with a link to all cpp files to compile
set(SOURCES bench_mass.cpp test_micro.cpp test_gemm.cpp)

# Add a different executable for each cpp file in SOURCES
foreach(sourcefile ${SOURCES})
    # I used a simple string replace, to cut off .cpp.
    string(REPLACE ".cpp" "" name ${sourcefile})
    add_executable(${name} ${sourcefile})
    target_include_directories(${name} PUBLIC ${XSMM_INCLUDES} ${FASTOR_INCLUDE_DIR})
    if (likwid_FOUND)
        target_link_libraries(${name} likwid::likwid ${BLAS_LIBRARIES} MPI::MPI_CXX)
    else()
        target_link_libraries(${name} ${BLAS_LIBRARIES} MPI::MPI_CXX)
    endif()
endforeach(sourcefile ${SOURCES})

# print used CMAKE_CXX_FLAGS
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")