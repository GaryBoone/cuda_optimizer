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

__global__ void EuclideanDistanceStridedKernel(int n, float2 *x, float2 *y,
                                               float *distance);

class EuclideanDistanceStrided : public IKernel<DistKernelFunc> {
 public:
  EuclideanDistanceStrided(int mnb, int mbs)
      : max_num_blocks_(mnb), max_block_size_(mbs) {}
  KernelInfo GetKernelInfo() const override {
    // For a length n Euclidean distance calculation:
    // Problem size: n, the number of distance calculations
    // Bandwidth: (2 reads from x and y + 1 write to distance) * n *
    // sizeof(float2)
    //            + (1 write to distance) * n * sizeof(float)
    return {"EuclideanDistanceStrided", n_,
            (3 * n_ * sizeof(float2)) + (n_ * sizeof(float))};
  }
  void (*GetKernel() const)(int, float2 *, float2 *, float *) override {
    return EuclideanDistanceStridedKernel;
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

}  // namespace cuda_optimizer
