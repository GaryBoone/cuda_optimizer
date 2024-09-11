#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels.h"
#include "matrix_multiply.h"

namespace cuda_optimizer {

void MatrixMultiply::Setup() {
  size_t size = n_ * sizeof(float);
  cudaMallocManaged(&a_, size);
  cudaMallocManaged(&b_, size);
  cudaMallocManaged(&c_, size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA allocation error: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Initialize matrices with random values.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < n_; i++) {
    a_[i] = dis(gen);
    b_[i] = dis(gen);
    c_[i] = 0.0f;
  }
}

void MatrixMultiply::RunKernel(int num_blocks, int block_size) {
  MatrixMultiplyKernel<<<num_blocks, block_size>>>(matrix_dim_, a_, b_, c_);
  cudaDeviceSynchronize();
}

void MatrixMultiply::Cleanup() {
  cudaFree(a_);
  cudaFree(b_);
  cudaFree(c_);
}

int MatrixMultiply::CheckResults() {
  // Compute CPU version for comparison.
  std::vector<float> C_cpu(n_, 0.0f);
  for (int i = 0; i < matrix_dim_; i++) {
    for (int j = 0; j < matrix_dim_; j++) {
      for (int k = 0; k < matrix_dim_; k++) {
        C_cpu[i * matrix_dim_ + j] +=
            a_[i * matrix_dim_ + k] * b_[k * matrix_dim_ + j];
      }
    }
  }

  int num_errors = 0;
  double max_error = 0.0;
  // Compare GPU results with CPU results.
  for (int i = 0; i < n_; i++) {
    float diff = fabs(c_[i] - C_cpu[i]);
    if (diff > 1e-3) {
      num_errors++;
      max_error = fmax(max_error, diff);
    }
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
