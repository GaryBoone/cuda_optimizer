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

class AddStridedManaged : public IKernel<AddKernelFunc> {
 public:
  AddStridedManaged(int mnb, int mbs)
      : max_num_blocks_(mnb), max_block_size_(mbs) {}
  KernelInfo GetKernelInfo() const override {
    // For a length n vector sum:
    // Problem size: n, the number of add operations
    // Bandwidth: (2 reads + 1 write) * n * sizeof(float)
    return {"AddStridedManaged", n_, 3 * n_ * sizeof(float)};
  }
  void (*GetKernel() const)(int, float *, float *) override {
    return AddStridedKernel;
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
  int n_ = 1 << 20;
  float *x_, *y_;
  const int max_num_blocks_;
  const int max_block_size_;
};
