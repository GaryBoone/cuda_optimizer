#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../example.h"
#include "../kernels.h"

__global__ void EuclidianDistanceUnstridedKernel(int n, float2 *x, float2 *y,
                                                 float *distance);

class EuclidianDistanceUnstrided
    : public IKernel<void (*)(int, float2 *, float2 *, float *)> {
 public:
  EuclidianDistanceUnstrided(int mnb, int mbs)
      : max_num_blocks_(mnb), max_block_size_(mbs) {}
  KernelInfo GetKernelInfo() const override {
    return {"EuclidianDistanceUnstrided", n_,
            sizeof(float2) * 2 + sizeof(float)};
  }
  void (*GetKernel() const)(int, float2 *, float2 *, float *) override {
    return EuclidianDistanceUnstridedKernel;
  }
  void Setup() override;
  std::unique_ptr<IGridSizeGenerator> GetNumBlocksGenerator() const override {
    return std::make_unique<DoublingGenerator>(max_num_blocks_);
  }
  std::unique_ptr<IGridSizeGenerator> GetBlockSizeGenerator() const override {
    return std::make_unique<IncrementBy32Generator>(max_block_size_);
  }
  void RunKernel(int num_blocks, int block_size) override;
  int CheckResults() override;
  void Cleanup() override;

 private:
  const int n_ = 1 << 20;
  float2 *d_x_, *d_y_;
  float *d_distance_;
  std::vector<float2> h_x_, h_y_;
  std::vector<float> h_distance_;
  const float tolerance_ = 1e-4;
  int max_num_blocks_;
  int max_block_size_;
};
