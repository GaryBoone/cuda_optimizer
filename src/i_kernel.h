#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "./generators.h"
#include "./kernels.h"

namespace cuda_optimizer {

struct KernelInfo {
  // name is the name of the kernel.
  std::string name;
  // n is the problem size, the total number of items that will be processed.
  int n;
  // total_bytes is the total number of bytes that will be transferred to the
  // GPU. This value is divided by the time it takes to transfer the data to
  // calculate the bandwidth.
  int total_bytes;

  KernelInfo(const std::string& kernel_name, int problem_size, size_t bytes)
      : name(kernel_name),
        n(problem_size),
        total_bytes(static_cast<int>(bytes)) {}
};

template <typename KernelFunc>
class IKernel {
 public:
  virtual KernelInfo GetKernelInfo() const = 0;
  virtual KernelFunc GetKernel() const = 0;
  virtual void Setup() = 0;
  virtual std::unique_ptr<IGridSizeGenerator> GetNumBlocksGenerator() const = 0;
  virtual std::unique_ptr<IGridSizeGenerator> GetBlockSizeGenerator() const = 0;
  virtual void RunKernel(int num_blocks, int block_size) = 0;
  virtual int CheckResults() = 0;
  virtual void Cleanup() = 0;

  void Run(int num_blocks, int block_size) {
    Setup();
    RunKernel(num_blocks, block_size);
    if (0 == CheckResults()) {
      std::cout << "    Results are correct" << std::endl << std::flush;
    } else {
      std::cout << "    Results are incorrect" << std::endl << std::flush;
    }
    Cleanup();
  }
};

}  // namespace cuda_optimizer