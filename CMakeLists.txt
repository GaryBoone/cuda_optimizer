cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cuda_optimize_project LANGUAGES CXX CUDA)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
