#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/format_number.h"

TEST_CASE("Format examples", "[format]") {
  REQUIRE(FormatNumber(-22) == "-22.00");
  REQUIRE(FormatNumber(0) == "0.00");
  REQUIRE(FormatNumber(-6e-6) == "-0.00");
  REQUIRE(FormatNumber(0.9e-2) == "0.01");
  REQUIRE(FormatNumber(1e-2) == "0.01");
  REQUIRE(FormatNumber(22) == "22.00");
  REQUIRE(FormatNumber(3e3) == "3.00K");
  REQUIRE(FormatNumber(2e6) == "2.00M");
  REQUIRE(FormatNumber(6.235e9) == "6.24G");
  REQUIRE(FormatNumber(6.235e10) == "62.35G");
  REQUIRE(FormatNumber(6.235e11) == "623.50G");
  REQUIRE(FormatNumber(6.235e12) == "6.24T");
  REQUIRE(FormatNumber(1.2e13) == "12.00T");
  REQUIRE(FormatNumber(1.5e15) == "1.50P");
  REQUIRE(FormatNumber(1.8e18) == "1.80E");
  REQUIRE(FormatNumber(2.1e21) == "2.10Z");
  REQUIRE(FormatNumber(2.4e24) == "2.40Y");
  REQUIRE(FormatNumber(2.8e27) == "2800.00Y");
}
