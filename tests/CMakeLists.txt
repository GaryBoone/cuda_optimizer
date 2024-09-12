include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.0)
FetchContent_MakeAvailable(catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/contrib")

set(TEST_FILES test_adaptive_sampler.cpp test_reporter.cpp test_metrics.cpp
               test_generators.cpp)

add_executable(test_app ${TEST_FILES})
target_link_libraries(test_app PRIVATE Catch2::Catch2WithMain core_lib)

include(CTest)
include(Catch)
catch_discover_tests(test_app)

# Note: ctest must be run from the build directory. So from the project
# directory: $ cmake --build build $ (cd build; ctest)
