
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

################################################################################
# CUDA
find_package(CUDA REQUIRED)  # Find the CUDA toolkit
include_directories(${CUDA_INCLUDE_DIRS})

# ################################################################################
# # `expected` header-only library
# include(FetchContent)
# FetchContent_Declare(
#   expected
#   GIT_REPOSITORY https://github.com/TartanLlama/expected.git
#   GIT_TAG        v1.1.0
# )
# # Disable building tests which generate some warnings.
# set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "Build tests for the expected library" FORCE)
# FetchContent_MakeAvailable(expected)
# include_directories(${expected_SOURCE_DIR}/include)

################################################################################
# cuda_adaptive
set(CPP_SOURCES
    adaptive_sampler.cpp
)

add_executable(cuda_optimizer cuda_optimizer.cu ${CPP_SOURCES} )

target_link_libraries(cuda_optimizer PRIVATE ${CUDA_LIBRARIES} expected_interface)

# target_include_directories(cuda_optimizer PRIVATE tl::expected)
# target_link_libraries(cuda_optimizer PRIVATE ${CUDA_LIBRARIES} tl::expected)

# target_include_directories(cuda_optimizer PRIVATE ${CUDA_INCLUDE_DIRS} ${expected_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(cuda_optimizer PRIVATE ${CUDA_LIBRARIES} tl::expected)
