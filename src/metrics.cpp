#include "./metrics.h"

#include <iostream>

namespace cuda_optimizer {

void Metrics::Update(Condition cond, const Data& new_metrics) {
  auto& current = data[static_cast<size_t>(cond)];
  if (IsBetter(new_metrics, current, cond)) {
    current = new_metrics;
  }
}

void Metrics::UpdateAll(const Data& new_data) {
  for (size_t i = 0; i < static_cast<size_t>(Condition::kLastCondition); ++i) {
    Update(static_cast<Condition>(i), new_data);
  }
}

bool Metrics::IsBetter(const Data& new_metrics, const Data& current,
                       Condition cond) const {
  switch (cond) {
    case Condition::kMinTime:
      if (new_metrics.time_ms < current.time_ms) return true;
      if (new_metrics.time_ms > current.time_ms) return false;
      return new_metrics.bandwidth > current.bandwidth ||
             (new_metrics.bandwidth == current.bandwidth &&
              new_metrics.occupancy > current.occupancy);

    case Condition::kMaxBandwidth:
      if (new_metrics.bandwidth > current.bandwidth) return true;
      if (new_metrics.bandwidth < current.bandwidth) return false;
      return new_metrics.time_ms < current.time_ms ||
             (new_metrics.time_ms == current.time_ms &&
              new_metrics.occupancy > current.occupancy);

    case Condition::kMaxOccupancy:
      if (new_metrics.occupancy > current.occupancy) return true;
      if (new_metrics.occupancy < current.occupancy) return false;
      return new_metrics.bandwidth > current.bandwidth ||
             (new_metrics.time_ms < current.time_ms &&
              new_metrics.bandwidth == current.bandwidth);

    default:
      return false;
  }
}

constexpr std::string_view Metrics::ConditionToString(Condition cond) const {
  switch (cond) {
    case Condition::kMinTime:
      return "Minimum Time";
    case Condition::kMaxBandwidth:
      return "Maximum Bandwidth";
    case Condition::kMaxOccupancy:
      return "Maximum Occupancy";
    default:
      return "Unknown";
  }
}

void Metrics::print_all() const {
  for (size_t i = 0; i < static_cast<size_t>(Condition::kLastCondition); ++i) {
    Condition cond = static_cast<Condition>(i);
    const auto& metrics = data[i];

    std::cout << "Condition: " << ConditionToString(cond) << "\n"
              << "  Time (ms): " << metrics.time_ms << "\n"
              << "  Bandwidth: " << metrics.bandwidth << "\n"
              << "  Num Blocks: " << metrics.num_blocks << "\n"
              << "  Block Size: " << metrics.block_size << "\n"
              << "  Occupancy: " << metrics.occupancy << "\n\n";
  }
}

}  // namespace cuda_optimizer