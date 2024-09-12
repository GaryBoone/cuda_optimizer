#pragma once

#include <stddef.h>

#include <array>
#include <limits>
#include <string_view>

namespace cuda_optimizer {

enum class Condition {
  kMinTime,
  kMaxBandwidth,
  kMaxOccupancy,
  kLastCondition  // Marker to count conditions.
};

struct Data {
  int num_blocks = 0;
  int block_size = 0;
  double time_ms = 0;
  double bandwidth = 0;
  double occupancy = 0;
};

class Metrics {
 public:
  Metrics() {
    (*this)[Condition::kMinTime].time_ms = std::numeric_limits<double>::max();
    (*this)[Condition::kMaxBandwidth].bandwidth =
        std::numeric_limits<double>::lowest();
    (*this)[Condition::kMaxOccupancy].occupancy =
        std::numeric_limits<double>::lowest();
  }

  // Update metrics for a given condition.
  void Update(Condition cond, const Data& new_metrics);

  // Update metrics for all conditions.
  void UpdateAll(const Data& new_data);

  // For ties on the primary condition, prefer the others in a given order.
  bool IsBetter(const Data& new_metrics, const Data& current,
                Condition cond) const;

  // Get metrics for a given condition
  const Data& get_metrics(Condition cond) const {
    return data[static_cast<size_t>(cond)];
  }

  void print_all() const;

  Data& operator[](Condition c) { return data[static_cast<size_t>(c)]; }

 private:
  std::array<Data, static_cast<size_t>(Condition::kLastCondition)> data;

  constexpr std::string_view ConditionToString(Condition cond) const;
};

}  // namespace cuda_optimizer
