#pragma once

#include <cuda_runtime.h>

#include "./adaptive_sampler.h"
#include "./i_kernel.h"

namespace cuda_optimizer {

template <typename KernelFunc>
double Occupancy(cudaDeviceProp props, int num_blocks, int block_size,
                 KernelFunc kernel) {
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, block_size,
                                                0);
  int activeWarps = num_blocks * block_size / props.warpSize;
  assert(0 != props.warpSize);
  int maxWarps = props.maxThreadsPerMultiProcessor / props.warpSize;
  return (static_cast<double>(activeWarps) / maxWarps);
}

// Calculate the optimimal num_blocks and block_size for the given kernel on
// the given hardware.
template <typename KernelFunc>
void OptimizeOccupancy(cudaDeviceProp &hardware_info, int &num_blocks,
                       int &block_size, KernelFunc kernel) {
  int min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

  int num_blocks_per_SM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_SM, kernel,
                                                block_size, 0);

  int num_SMs = hardware_info.multiProcessorCount;
  num_blocks = num_blocks_per_SM * num_SMs;

  double current_occupancy =
      Occupancy(hardware_info, num_blocks, block_size, kernel);

  for (int bs = block_size; bs >= 32; bs -= 32) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_SM, kernel,
                                                  bs, 0);
    int nb = num_blocks_per_SM * num_SMs;
    double occ = Occupancy(hardware_info, nb, bs, kernel);

    if (occ > current_occupancy) {
      num_blocks = nb;
      block_size = bs;
      current_occupancy = occ;
    }

    if (current_occupancy >= 0.99) break;  // Close enough to 1.0
  }
}

}  // namespace cuda_optimizer
