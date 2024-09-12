#include <catch2/catch_test_macros.hpp>
#include <limits>

#include "../src/metrics.h"

namespace cuda_optimizer {

TEST_CASE("Metrics::Initialization", "[Metrics]") {
  Metrics m;
  REQUIRE(m[Condition::kMinTime].time_ms == std::numeric_limits<double>::max());
  REQUIRE(m[Condition::kMinTime].bandwidth == 0.0);
  REQUIRE(m[Condition::kMinTime].occupancy == 0.0);

  REQUIRE(m[Condition::kMaxBandwidth].time_ms == 0.0);
  REQUIRE(m[Condition::kMaxBandwidth].bandwidth ==
          std::numeric_limits<double>::lowest());
  REQUIRE(m[Condition::kMaxBandwidth].occupancy == 0.0);

  REQUIRE(m[Condition::kMaxOccupancy].time_ms == 0.0);
  REQUIRE(m[Condition::kMaxOccupancy].bandwidth == 0.0);
  REQUIRE(m[Condition::kMaxOccupancy].occupancy ==
          std::numeric_limits<double>::lowest());
}

TEST_CASE("Metrics::Update updates correctly", "[metrics]") {
  Metrics metrics;

  SECTION("Update for kMinTime condition") {
    // nb, bs, t, bw, o
    Data new_data{100, 32, 10, 1024, 0.75};
    metrics.Update(Condition::kMinTime, new_data);

    REQUIRE(metrics[Condition::kMinTime].num_blocks == 100);
    REQUIRE(metrics[Condition::kMinTime].block_size == 32);
    REQUIRE(metrics[Condition::kMinTime].time_ms == 10.0);
    REQUIRE(metrics[Condition::kMinTime].bandwidth == 1024.0);
    REQUIRE(metrics[Condition::kMinTime].occupancy == 0.75);

    Data better_time{100, 32, 9, 1024, 0.70};
    metrics.Update(Condition::kMinTime, better_time);
    REQUIRE(metrics[Condition::kMinTime].time_ms == 9.0);
  }

  SECTION("Update for kMaxBandwidth condition") {
    Data new_data{100, 32, 10, 1024, 0.75};
    metrics.Update(Condition::kMaxBandwidth, new_data);
    REQUIRE(metrics[Condition::kMaxBandwidth].bandwidth == 1024);
    Data better_bandwidth{100, 32, 10, 1025, 0.75};
    metrics.Update(Condition::kMaxBandwidth, better_bandwidth);
    REQUIRE(metrics[Condition::kMaxBandwidth].bandwidth == 1025);
  }

  SECTION("Update for kMaxOccupancy condition") {
    Data new_data{100, 32, 10, 1024, 0.75};
    metrics.Update(Condition::kMaxOccupancy, new_data);
    REQUIRE(metrics[Condition::kMaxOccupancy].occupancy == 0.75);
    Data better_occupancy{100, 32, 10, 1024, 0.8};
    metrics.Update(Condition::kMaxOccupancy, better_occupancy);
    REQUIRE(metrics[Condition::kMaxOccupancy].occupancy == 0.8);
  }
}

TEST_CASE("Metrics::IsBetter compares correctly", "[metrics]") {
  Metrics metrics;

  SECTION("kMinTime condition") {
    Data current{100, 50, 10, 1024, 0.75};
    Data better_time{100, 50, 9, 1024, 0.75};
    Data worse_time{100, 50, 11, 1024, 0.75};
    Data equal_time_better_bandwidth{100, 50, 10, 1025, 0.75};

    REQUIRE(metrics.IsBetter(better_time, current, Condition::kMinTime));
    REQUIRE_FALSE(metrics.IsBetter(worse_time, current, Condition::kMinTime));
    REQUIRE(metrics.IsBetter(equal_time_better_bandwidth, current,
                             Condition::kMinTime));
  }

  SECTION("kMaxBandwidth condition") {
    Data current{100, 50, 10, 1024, 0.75};
    Data better_bandwidth{100, 50, 10, 1025, 0.75};
    Data worse_bandwidth{100, 50, 10, 1023, 0.75};
    Data equal_bandwidth_better_time{100, 50, 9, 1024, 0.75};

    REQUIRE(
        metrics.IsBetter(better_bandwidth, current, Condition::kMaxBandwidth));
    REQUIRE_FALSE(
        metrics.IsBetter(worse_bandwidth, current, Condition::kMaxBandwidth));
    REQUIRE(metrics.IsBetter(equal_bandwidth_better_time, current,
                             Condition::kMaxBandwidth));
  }

  SECTION("kMaxOccupancy condition") {
    Data current{100, 50, 10, 1024, 0.75};
    Data better_occupancy{100, 50, 10, 1024, 0.80};
    Data worse_occupancy{100, 50, 10, 1024, 0.70};
    Data equal_occupancy_better_bandwidth{100, 50, 10, 1025, 0.75};

    REQUIRE(
        metrics.IsBetter(better_occupancy, current, Condition::kMaxOccupancy));
    REQUIRE_FALSE(
        metrics.IsBetter(worse_occupancy, current, Condition::kMaxOccupancy));
    REQUIRE(metrics.IsBetter(equal_occupancy_better_bandwidth, current,
                             Condition::kMaxOccupancy));
  }
}

TEST_CASE("Metrics::UpdateAll updates all conditions", "[metrics]") {
  Metrics metrics;
  Data new_data{100, 50, 10, 1024, 0.75};

  metrics.UpdateAll(new_data);

  for (int i = 0; i < static_cast<int>(Condition::kLastCondition); ++i) {
    Condition cond = static_cast<Condition>(i);
    REQUIRE(metrics[cond].num_blocks == 100);
    REQUIRE(metrics[cond].block_size == 50);
    REQUIRE(metrics[cond].time_ms == 10.0);
    REQUIRE(metrics[cond].bandwidth == 1024);
    REQUIRE(metrics[cond].occupancy == 0.75);
  }
}

}  // namespace cuda_optimizer
