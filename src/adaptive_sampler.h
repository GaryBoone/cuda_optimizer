#pragma once

#include "./errors.h"

namespace cuda_optimizer {

class AdaptiveSampler {
  friend class AdaptiveSamplerTest;

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

}  // namespace cuda_optimizer
