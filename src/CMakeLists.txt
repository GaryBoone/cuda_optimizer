# ##############################################################################
# CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF) # No architecture restrictions.

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

# ##############################################################################
# Show the full compiler flags as they're run. set(CMAKE_VERBOSE_MAKEFILE ON
# CACHE BOOL "ON" FORCE)

# ##############################################################################
# `expected` header-only library
include(FetchContent)
FetchContent_Declare(
  expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG v1.1.0)

set(EXPECTED_BUILD_TESTS
    OFF
    CACHE BOOL "Build tests for the expected library" FORCE)
FetchContent_MakeAvailable(expected)

# ##############################################################################
# application library
set(SOURCE_FILES
    adaptive_sampler.cpp
    examples/add_strided_managed.cu
    examples/add_strided_unmanaged.cu
    examples/add_unstrided_managed.cu
    examples/add_unstrided_unmanaged.cu
    examples/euclidean_distance_strided.cu
    examples/euclidean_distance_unstrided.cu
    examples/matrix_multiply.cu
    kernels.cu
    metrics.cpp
    reporter.cpp)

add_library(core_lib ${SOURCE_FILES})
target_include_directories(core_lib PUBLIC ${expected_SOURCE_DIR}/include)

# ##############################################################################
# application executable
add_executable(cuda_optimizer cuda_optimizer.cu)
target_link_libraries(cuda_optimizer PRIVATE core_lib)
