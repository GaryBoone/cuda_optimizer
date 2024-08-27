
################################################################################
# CUDA
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)  # Find the CUDA toolkit
include_directories(${CUDA_INCLUDE_DIRS})

################################################################################
# `expected` header-only library
include(FetchContent)
FetchContent_Declare(
  expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG        v1.1.0
)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "Build tests for the expected library" FORCE)
FetchContent_MakeAvailable(expected)

################################################################################
# application library
add_library(core_lib adaptive_sampler.cpp)
target_include_directories(core_lib PUBLIC ${expected_SOURCE_DIR}/include)

################################################################################
# application executable
add_executable(cuda_optimizer cuda_optimizer.cu)
target_link_libraries(cuda_optimizer PRIVATE core_lib)
