#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "add_strided_managed.h"

namespace cuda_optimizer {

void AddStridedManaged::Setup() {
  cudaMallocManaged(&x_, n_ * sizeof(float));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA allocation error: " << cudaGetErrorString(err)
              << std::endl;
  }
  cudaMallocManaged(&y_, n_ * sizeof(float));
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "allocation error: " << cudaGetErrorString(err) << std::endl;
  }
  for (int j = 0; j < n_; j++) {
    x_[j] = 1.0f;
    y_[j] = 2.0f;
  }
}

void AddStridedManaged::RunKernel(int num_blocks, int block_size) {
  AddStridedKernel<<<num_blocks, block_size>>>(n_, x_, y_);
  cudaDeviceSynchronize();
}

void AddStridedManaged::Cleanup() {
  cudaFree(x_);
  cudaFree(y_);
}

int AddStridedManaged::CheckResults() {
  int num_errors = 0;
  double max_error = 0.0;

  for (int i = 0; i < n_; i++) {
    if (fabs(y_[i] - 3.0f) > 1e-6) {
      num_errors++;
    }
    max_error = fmax(max_error, fabs(y_[i] - 3.0f));
  }

  if (num_errors > 0) {
    std::cout << "  number of errors: " << num_errors;
  }
  if (max_error > 0.0) {
    std::cout << ",  max error: " << max_error;
  }

  return num_errors;
}

}  // namespace cuda_optimizer
