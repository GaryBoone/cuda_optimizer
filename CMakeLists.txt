# A CMake version equal to or greater than 3.18 is needed to work correctly with
# CMAKE_CUDA_STANDARD 17. This project was developed with CMake 3.29.8. See
# src/CMakeLists.txt.
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cuda_optimize_project LANGUAGES CXX CUDA)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
