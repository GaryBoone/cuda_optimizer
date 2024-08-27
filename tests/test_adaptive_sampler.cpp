#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "../src/adaptive_sampler.h"

TEST_CASE("AdaptiveSampler functionality is tested", "AdaptiveSampler") {
  AdaptiveSampler as;
  REQUIRE(as.should_continue() == true);
}
