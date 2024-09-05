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

struct KernelInfo {
  std::string name;
  int n;
  int bytesPerElement;
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
      std::cout << "  Results are correct" << std::endl << std::flush;
    } else {
      std::cout << "  Results are incorrect" << std::endl << std::flush;
    }
    Cleanup();
  }
};
