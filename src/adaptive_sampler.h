#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "errors.h"

class AdaptiveSampler {
  friend class AdaptiveSamplerTest; // Make the test class a friend

private:
  double alpha_ = 0.0;
  int num_samples_ = 0;
  double relative_precision_;

  ExpectedDouble TwoTailed95PercentStudentsT(int df);

public:
  explicit AdaptiveSampler(double rp = 0.30) : relative_precision_(rp) {}

  void Update(double x);
  bool ShouldContinue();
  ExpectedDouble EstimatedMean();
  int NumSamples() { return num_samples_; }
};
