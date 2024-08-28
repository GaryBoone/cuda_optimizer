#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/format_number.h"

TEST_CASE("Format examples", "[format]") {
  REQUIRE(formatNumber(-22) == "-22.00");
  REQUIRE(formatNumber(0) == "0.00");
  REQUIRE(formatNumber(-6e-6) == "-0.00");
  REQUIRE(formatNumber(0.9e-2) == "0.01");
  REQUIRE(formatNumber(1e-2) == "0.01");
  REQUIRE(formatNumber(22) == "22.00");
  REQUIRE(formatNumber(3e3) == "3.00K");
  REQUIRE(formatNumber(2e6) == "2.00M");
  REQUIRE(formatNumber(6.235e9) == "6.24G");
  REQUIRE(formatNumber(6.235e10) == "62.35G");
  REQUIRE(formatNumber(6.235e11) == "623.50G");
  REQUIRE(formatNumber(6.235e12) == "6.24T");
  REQUIRE(formatNumber(1.2e13) == "12.00T");
  REQUIRE(formatNumber(1.5e15) == "1.50P");
  REQUIRE(formatNumber(1.8e18) == "1.80E");
  REQUIRE(formatNumber(2.1e21) == "2.10Z");
  REQUIRE(formatNumber(2.4e24) == "2.40Y");
  REQUIRE(formatNumber(2.8e27) == "2800.00Y");
}
