cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "tcfem")
project(${PROJECT_NAME} VERSION 1.0.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Wall -Wextra -pedantic")

include_directories("src")

# Find likwid
find_package(likwid)

# Find BLAS (any flavor)
find_package(BLAS REQUIRED)

# find MPI
find_package(MPI REQUIRED)

# find libxsmm
if(NOT DEFINED ENV{LIBXSMM_HOME})
    message(STATUS "LIBXSMM_HOME is not defined")
else()
    set(XSMM_INCLUDES $ENV{LIBXSMM_HOME}/include)
    message(STATUS "LIBXSMM_HOME is ${XSMM_INCLUDES}")
endif()

if (DEFINED ENV{LIKWID_PERFMON})
    message(STATUS "LIKWID_PERFMON is defined")
    add_definitions(-DLIKWID_PERFMON)
endif()

# Create a sources variable with a link to all cpp files to compile
set(SOURCES bench_mass.cpp)

# Add a different executable for each cpp file in SOURCES
foreach(sourcefile ${SOURCES})
    # I used a simple string replace, to cut off .cpp.
    string(REPLACE ".cpp" "" name ${sourcefile})
    add_executable(${name} ${sourcefile})
    target_include_directories(${name} PUBLIC ${XSMM_INCLUDES} ${LIKWID_INCLUDE_DIR})
    if (likwid_FOUND)
        target_link_libraries(${name} likwid::likwid ${BLAS_LIBRARIES} MPI::MPI_CXX)
    else()
        target_link_libraries(${name} ${BLAS_LIBRARIES} MPI::MPI_CXX)
    endif()
endforeach(sourcefile ${SOURCES})

# print used CMAKE_CXX_FLAGS
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
