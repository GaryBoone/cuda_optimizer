#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels.h"
#include "euclidean_distance_unstrided.h"

namespace cuda_optimizer {

void EuclideanDistanceUnstrided::Setup() {
  h_x_.resize(n_);
  h_y_.resize(n_);
  h_distance_.resize(n_);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);

  for (int i = 0; i < n_; ++i) {
    h_x_[i] = make_float2(dis(gen), dis(gen));
    h_y_[i] = make_float2(dis(gen), dis(gen));
  }

  cudaMalloc(&d_x_, n_ * sizeof(float2));
  cudaMalloc(&d_y_, n_ * sizeof(float2));
  cudaMalloc(&d_distance_, n_ * sizeof(float));

  cudaMemcpy(d_x_, h_x_.data(), n_ * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_, h_y_.data(), n_ * sizeof(float2), cudaMemcpyHostToDevice);
}

void EuclideanDistanceUnstrided::RunKernel(int num_blocks, int block_size) {
  EuclideanDistanceUnstridedKernel<<<num_blocks, block_size>>>(n_, d_x_, d_y_,
                                                               d_distance_);
  cudaDeviceSynchronize();

  cudaMemcpy(h_distance_.data(), d_distance_, n_ * sizeof(float),
             cudaMemcpyDeviceToHost);
}

void EuclideanDistanceUnstrided::Cleanup() {
  cudaFree(d_x_);
  cudaFree(d_y_);
  cudaFree(d_distance_);
}

int EuclideanDistanceUnstrided::CheckResults() {
  int num_errors = 0;
  float max_error = 0.0;
  for (int i = 0; i < n_; ++i) {
    float dx = h_y_[i].x - h_x_[i].x;
    float dy = h_y_[i].y - h_x_[i].y;
    float expected = std::sqrt(dx * dx + dy * dy);
    float error = std::abs(h_distance_[i] - expected);
    if (std::abs(h_distance_[i] - expected) > tolerance_) {
      num_errors++;
      max_error = fmax(max_error, error);
    }
  }
  if (num_errors > 0) {
    std::cout << "  errors found: number of errors: " << num_errors;
  }
  if (max_error > 0.0) {
    std::cout << ",  max error: " << max_error;
  }
  if (num_errors > 0 || max_error > 0.0) {
    std::cout << std::endl;
  }
  return num_errors;
}

}  // namespace cuda_optimizer
