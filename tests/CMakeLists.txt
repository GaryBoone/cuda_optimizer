include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7  # specify the version of Catch2
)
FetchContent_MakeAvailable(catch2)

# Setup testing executable
add_executable(test_app test_adaptive_sampler.cpp ../src/adaptive_sampler.cpp)
target_link_libraries(test_app PRIVATE Catch2::Catch2 expected_interface)

# Register tests
include(CTest)
# include(Catch)
# catch_discover_tests(test_app)
