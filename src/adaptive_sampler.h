#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "errors.h"

class AdaptiveSampler {
private:
  double alpha = 0.0;
  int num_samples = 0;
  double relative_precision;

  ExpectedDouble two_tailed_95_students_t(int df);

public:
  explicit AdaptiveSampler(double rp = 0.30) : relative_precision(rp) {}

  void update(double x);
  bool should_continue();
  ExpectedDouble get_estimate();
  int get_num_samples() { return num_samples; }
};
