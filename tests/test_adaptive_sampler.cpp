#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include <catch2/catch.hpp>

#include "../src/adaptive_sampler.h"

TEST_CASE("AdaptiveSampler functionality is tested", "AdaptiveSampler") {
  AdaptiveSampler as;
  REQUIRE(as.should_continue() == true);
}
