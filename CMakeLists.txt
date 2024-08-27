cmake_minimum_required(VERSION 3.16)

project(cuda_optimize_project LANGUAGES CXX CUDA)

# Add the main application code
add_subdirectory(src)

# Add the test code
enable_testing()
add_subdirectory(tests)

################################################################################
# `expected` header-only library
include(FetchContent)
FetchContent_Declare(
  expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG        v1.1.0
)
FetchContent_MakeAvailable(expected)

# Create an interface library for expected
add_library(expected_interface INTERFACE)
target_include_directories(expected_interface INTERFACE ${expected_SOURCE_DIR}/include)
