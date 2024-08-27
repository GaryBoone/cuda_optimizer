#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/adaptive_sampler.h"

class AdaptiveSamplerTest {
public:
  static ExpectedDouble
  invoke_two_tailed_95_students_t(AdaptiveSampler &sampler, int df) {
    return sampler.two_tailed_95_students_t(df);
  }
};

TEST_CASE("AdaptiveSampler initialization", "[AdaptiveSampler]") {
  AdaptiveSampler as;
  REQUIRE(as.should_continue());
  REQUIRE(as.get_num_samples() == 0);
  auto result = as.get_estimate();
  REQUIRE_FALSE(result.has_value());
  REQUIRE(result.error().error_type == ErrorInfo::TooFewSamples);
  REQUIRE(result.error().message ==
          "number of samples must be greater than zero");
}

TEST_CASE("AdaptiveSampler updating samples", "[AdaptiveSampler]") {
  AdaptiveSampler as;

  SECTION("One samples") {
    as.update(1.0);

    REQUIRE(as.should_continue());
    REQUIRE(as.get_num_samples() == 1);
    auto result = as.get_estimate();
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.0, 1e-10));
  }

  SECTION("Two samples") {
    as.update(1.4);
    as.update(1.2);

    REQUIRE(as.should_continue());
    REQUIRE(as.get_num_samples() == 2);
    auto result = as.get_estimate();
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.3, 1e-10));
  }

  SECTION("Three samples") {
    as.update(1.4);
    as.update(1.2);
    as.update(2.2);

    REQUIRE(as.should_continue());
    REQUIRE(as.get_num_samples() == 3);
    auto result = as.get_estimate();
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.6, 1e-10));
  }
}

TEST_CASE("two_tailed_95_students_t tests", "[AdaptiveSampler]") {
  AdaptiveSampler sampler;

  SECTION("Degrees of freedom is 0") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 0);
    REQUIRE(result.error().error_type == ErrorInfo::InvalidDegreesOfFreedom);
  }

  SECTION("Degrees of freedom is 1") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 1);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(12.706, 1e-12));
  }

  SECTION("Degrees of freedom is 10") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 10);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(2.228, 1e-12));
  }

  SECTION("Degrees of freedom is 55") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 55);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(2.004, 1e-12));
  }

  SECTION("Degrees of freedom is 355") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 355);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.9669, 1e-12));
  }

  SECTION("Degrees of freedom is 1355") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 1355);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.962, 1e-12));
  }

  SECTION("Degrees of freedom is 21355") {
    auto result =
        AdaptiveSamplerTest::invoke_two_tailed_95_students_t(sampler, 21355);
    REQUIRE(result.has_value());
    REQUIRE_THAT(result.value(), Catch::Matchers::WithinAbs(1.962, 1e-12));
  }
}

TEST_CASE("AdaptiveSampler should continue", "[AdaptiveSampler]") {
  AdaptiveSampler as;

  SECTION("No samples") { REQUIRE(as.should_continue()); }
  SECTION("One samples") {
    as.update(1.0);
    REQUIRE(as.should_continue());
  }
  SECTION("Two samples") {
    as.update(1.0);
    as.update(12.0);
    REQUIRE(as.should_continue());
  }
  SECTION("Three samples") {
    as.update(1.0);
    as.update(12.0);
    as.update(4.0);
    REQUIRE(as.should_continue());
  }
  SECTION("52 samples is enough") {
    as.update(6.11);
    as.update(6.09);
    for (int i = 0; i < 40; i++) {
      as.update(6.10);
    }
    REQUIRE(as.should_continue());

    for (int i = 0; i < 10; i++) {
      as.update(6.10);
    }
    REQUIRE(!as.should_continue());
  }
}
