include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.7.0
)
FetchContent_MakeAvailable(catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/contrib")

add_executable(test_app test_adaptive_sampler.cpp)
target_link_libraries(test_app PRIVATE Catch2::Catch2 core_lib)

include(CTest)
include(Catch)
catch_discover_tests(test_app)

# Note: ctest must be run from the build directory. So from the project
# directory:
#     $ (cd build; ctest)
