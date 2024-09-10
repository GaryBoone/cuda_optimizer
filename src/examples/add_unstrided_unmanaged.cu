#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "add_unstrided_unmanaged.h"

void AddUnstridedUnmanaged::Setup() {
  // Allocate memory on the device.
  cudaMalloc(&x_, n_ * sizeof(float));
  cudaMalloc(&y_, n_ * sizeof(float));

  h_x_ = new float[n_];
  h_y_ = new float[n_];

  for (int j = 0; j < n_; j++) {
    h_x_[j] = 1.0f;
    h_y_[j] = 2.0f;
  }

  // Copy data from host to device.
  cudaMemcpy(x_, h_x_, n_ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_, h_y_, n_ * sizeof(float), cudaMemcpyHostToDevice);

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
}

void AddUnstridedUnmanaged::RunKernel(int num_blocks, int block_size) {
  AddUnstridedKernel<<<num_blocks, block_size>>>(n_, x_, y_);
  cudaDeviceSynchronize();

  // Copy results back to host.
  cudaMemcpy(h_y_, y_, n_ * sizeof(float), cudaMemcpyDeviceToHost);
}

void AddUnstridedUnmanaged::Cleanup() {
  cudaFree(x_);
  cudaFree(y_);
  delete[] h_x_;
  delete[] h_y_;
}

int AddUnstridedUnmanaged::CheckResults() {
  int num_errors = 0;
  double max_error = 0.0;

  for (int i = 0; i < n_; i++) {
    if (fabs(h_y_[i] - 3.0f) > 1e-6) {
      num_errors++;
    }
    max_error = fmax(max_error, fabs(h_y_[i] - 3.0f));
  }

  if (num_errors > 0) {
    std::cout << "  number of errors: " << num_errors;
  }
  if (max_error > 0.0) {
    std::cout << ",  max error: " << max_error;
  }

  return num_errors;
}
