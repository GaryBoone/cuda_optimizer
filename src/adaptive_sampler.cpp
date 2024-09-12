#include "adaptive_sampler.h"

#include <stddef.h>

#include <cmath>
#include <utility>
#include <vector>

#include "./errors.h"
#include "tl/expected.hpp"

namespace cuda_optimizer {

// Calculate the student's t distribution value for the 2-sided 95% confidence
// interval for the given degrees of freedom. This function assumes that the
// caller has already reduced the degrees of freedom, if needed.
ExpectedDouble AdaptiveSampler::TwoTailed95PercentStudentsT(int df) {
  if (df <= 0) {
    return tl::make_unexpected(
        ErrorInfo(ErrorInfo::kInvalidDegreesOfFreedom,
                  "degrees of freedom must be positive"));
  }
  // Table values for degrees of freedom 1--40.
  const std::vector<double> table_values = {
      12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
      2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
      2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
      2.040,  2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021};

  if (df <= 40) {
    return table_values[df - 1];
  }

  // Additional interpolation points.
  const std::vector<std::pair<int, double>> interpolation_points = {
      {45, 2.014},  {50, 2.009},  {55, 2.004},  {60, 2.000},  {70, 1.994},
      {80, 1.990},  {90, 1.987},  {100, 1.984}, {150, 1.976}, {200, 1.972},
      {250, 1.969}, {300, 1.968}, {400, 1.966}, {500, 1.965}, {600, 1.964},
      {700, 1.963}, {800, 1.963}, {900, 1.963}, {1000, 1.962}};

  auto interpolate = [](int x1, double y1, int x2, double y2, int x) {
    // In all cases in the table, x2 > x1, so no division by 0 risk.
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
  };

  // Find the appropriate interpolation range and interpolate.
  for (size_t i = 0; i < interpolation_points.size() - 1; ++i) {
    if (df <= interpolation_points[i + 1].first) {
      return interpolate(interpolation_points[i].first,
                         interpolation_points[i].second,
                         interpolation_points[i + 1].first,
                         interpolation_points[i + 1].second, df);
    }
  }
  return interpolation_points.back().second;
}

void AdaptiveSampler::Update(double x) {
  alpha_ += x;
  num_samples_++;
}

ExpectedDouble AdaptiveSampler::EstimatedMean() {
  if (0 == num_samples_) {
    return tl::make_unexpected(
        ErrorInfo(ErrorInfo::kTooFewSamples,
                  "number of samples must be greater than zero"));
  }
  return alpha_ / static_cast<double>(num_samples_);
}

// Returns true while more data is needed for an accurate estimate of the data's
// mean.
//
// Details:
// • The estimate of the mean has an error that shrinks as more data is
// gathered.
// • The standard deviation serves as a measure of data variability.
// • We want to gather enough data to be sure that our error is small relative
//   to the variance of the data.
// So we continue adding data while the error is larger than a
// `relative_precision_` of the standard deviation of the sample.
//
// For example, suppose we set the relative_precision_ to 0.3.
// Approximately 68% of the data is within one SD (±1 SD). If we
// linearly interpolate, we can estimate the coverage for 0.3 SD:
//     0.3 × 68%/2 = 0.3 × 34% ≈ 10.2%
// This means that approximately 10.2 % of the data lies between the mean and
// 0.3 SDs in one direction, or about 20.4% total within ±0.3 SDs. So if we set
// the relative_precision_ to 0.3, we'll add data to increase the estimate
// accuracy until we have 20.4% of the total data within ±0.3 SDs.

// Specifically, it calculates the margin of error (MoE) for the sample mean
// using a 95% confidence interval from the t-distribution and checks if this
// MoE exceeds the given `relative_precision_` of the standard deviation of the
// sample. The standard deviation serves as a measure of data variability.
//
// In summary, when this function returns false and we stop collecting data, we
// are 95% confident that the true mean lies within the margin of error
// calculated from the sample mean, which we set to be less than x% of the
// observed standard deviation of the sample.
bool AdaptiveSampler::ShouldContinue() {
  if (num_samples_ <= 1) {
    return true;
  }

  double beta = static_cast<double>(num_samples_);
  double mean = alpha_ / beta;
  double variance = alpha_ / (beta * beta * (beta - 1.0));

  // The standard error quantifies how much the sample mean is likely to vary
  // from the true population mean.
  double std_error = std::sqrt(variance / beta);
  // Use num_samples - 1 because the mean estimate reduces the degrees of
  // freedom. That is, if if we had the mean and n-1 values, we could
  // reconstruct the missing value. So we don't have n degrees of freedom in
  // this estimate, but rather one less.
  auto t_result = TwoTailed95PercentStudentsT(num_samples_ - 1);
  if (!t_result) {
    return true;
  }

  // The standard deviation quantifies the overall data variability.
  double std_dev = std::sqrt(variance);
  double margin_of_error = *t_result * std_error;
  double required_margin_of_error = relative_precision_ * std_dev;
  // The margin of error starts large and shrinks as we gather data because it's
  // proportional to sqrt(1/n). So keep sampling if the margin of error is
  // larger than the required margin of error.
  return margin_of_error > required_margin_of_error;
}

}  // namespace cuda_optimizer
