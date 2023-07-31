cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME "tcfem")
project(${PROJECT_NAME} VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")

include_directories("src")

# find google benchmark
find_package(benchmark REQUIRED)

add_executable(${PROJECT_NAME} permute.cpp)
target_link_libraries(${PROJECT_NAME} benchmark::benchmark)