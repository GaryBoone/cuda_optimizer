#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../i_kernel.h"
#include "../kernels.h"

namespace cuda_optimizer {

class MatrixMultiply : public IKernel<MatrixMultiplyKernelFunc> {
 public:
  MatrixMultiply(int mnb, int mbs)
      : max_num_blocks_(mnb), max_block_size_(mbs) {}
  KernelInfo GetKernelInfo() const override {
    // For an m x m matrix multiply:
    // Problem size: m² (number of multiply-add operations)
    // Bandwidth: (2 reads + 1 write) * m² * sizeof(float)
    int problem_size = matrix_dim_ * matrix_dim_;
    size_t total_bytes = 3 * matrix_dim_ * matrix_dim_ * sizeof(float);
    return {"MatrixMultiply", problem_size, total_bytes};
  }
  void (*GetKernel() const)(int, float *, float *, float *) override {
    return MatrixMultiplyKernel;
  }
  void Setup() override;
  std::unique_ptr<IGridSizeGenerator> GetNumBlocksGenerator() const override {
    return std::make_unique<DoublingGenerator>(max_num_blocks_);
  }
  std::unique_ptr<IGridSizeGenerator> GetBlockSizeGenerator() const override {
    return std::make_unique<IncrementBy32Generator>(max_block_size_);
  }
  void RunKernel(int num_blocks, int block_size) override;
  void Cleanup() override;
  int CheckResults() override;

 private:
  int matrix_dim_ = 1 << 7;  // 1 << 7 == 128
  // n_ is the overall problem size.
  int n_ = matrix_dim_ * matrix_dim_;  // (1 << 7)^2 == 1 << 14 == 16384
  float *a_, *b_, *c_;
  const int max_num_blocks_;
  const int max_block_size_;
};

}  // namespace cuda_optimizer
