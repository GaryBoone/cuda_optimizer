#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "adaptive_sampler.h"
#include "examples/add_strided_managed.h"
#include "examples/add_strided_unmanaged.h"
#include "examples/add_unstrided_managed.h"
#include "examples/add_unstrided_unmanaged.h"
#include "examples/euclidian_distance_strided.h"
#include "examples/euclidian_distance_unstrided.h"
#include "examples/matrix_multiply.h"
#include "grid_searchers.h"
#include "i_kernel.h"
#include "kernels.h"
#include "optimizer.h"
#include "timer.h"

// Build:
// $ cmake -B build -S .
// $ cmake --build build
// Run:
// $ ./build/src/cuda_optimizer
// Test:
// $ ./build/tests/test_app

template <typename KernelFunc>
float TimeKernel(IKernel<KernelFunc> &ex, int num_blocks, int block_size) {
  CudaTimer timer;
  timer.Start();
  ex.RunKernel(num_blocks, block_size);
  timer.Stop();

  // Wait for GPU to finish before accessing on host.
  cudaDeviceSynchronize();

  return timer.ElapsedMilliseconds();
}

cudaDeviceProp HardwareInfo() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);  // Get the number of devices
  if (num_devices == 0) {
    std::cout << "No CUDA devices found." << std::endl;
    exit(1);  // TODO(Gary): Fix.
    // return 0;
  }

  std::cout << "Number of CUDA devices: " << num_devices << std::endl;
  cudaDeviceProp props;
  for (int i = 0; i < num_devices; i++) {
    cudaGetDeviceProperties(&props, i);
    std::cout << "Device Number: " << i << std::endl;
    std::cout << "  Device name: " << props.name << std::endl;
    std::cout << "  Number of SMs: " << props.multiProcessorCount << std::endl;
    std::cout << "  Total global memory: "
              << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Compute capability: " << props.major << "." << props.minor
              << std::endl;
    std::cout << "  Maximum threads per SM: "
              << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Maximum warps: " << props.warpSize << std::endl;
    std::cout << "  Maximum threads per block: " << props.maxThreadsPerBlock
              << std::endl;
    std::cout << "  Maximum thread dimensions: (" << props.maxThreadsDim[0]
              << ", " << props.maxThreadsDim[1] << ", "
              << props.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Amount of shared memory per SM: "
              << props.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "  Number of registers per SM: " << props.regsPerMultiprocessor
              << std::endl;
  }
  return props;
}

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

template <typename KernelFunc>
tl::expected<AdaptiveSampler, ErrorInfo> RepeatUntil(double required_precision,
                                                     IKernel<KernelFunc> &ex,
                                                     int num_blocks,
                                                     int block_size) {
  AdaptiveSampler stats(required_precision);
  bool skip_first = true;
  while (stats.ShouldContinue()) {
    ex.Setup();

    float time = TimeKernel(ex, num_blocks, block_size);

    if (0 != ex.CheckResults()) {
      return tl::make_unexpected(ErrorInfo(ErrorInfo::kUnexpectedKernelResult,
                                           "errors in kernel results"));
    }
    ex.Cleanup();

    // Don't include the first run in the averages to ignore loading effects.
    if (skip_first) {
      skip_first = false;
      continue;
    }
    stats.Update(time);
  }
  return stats;
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

int main(void) {
  auto hardware_info = HardwareInfo();
  int max_num_blocks = hardware_info.maxThreadsDim[0] *
                       hardware_info.maxThreadsDim[1] *
                       hardware_info.maxThreadsDim[2];
  int max_block_size = hardware_info.maxThreadsPerBlock;
  std::cout << "...which means that: " << std::endl;
  std::cout << "  max_num_blocks: "
            << Reporter::FormatWithCommas(max_num_blocks) << std::endl;
  std::cout << "  max_block_size: "
            << Reporter::FormatWithCommas(max_block_size) << std::endl;

  // Individual runs.
  std::cout << "\n==> Add with stride kernel and with managed memory :"
            << std::endl;
  AddStridedManaged add_strided_managed(max_num_blocks, max_block_size);
  add_strided_managed.Run(4096, 256);

  std::cout << "\n==> Add without stride kernel and with managed memory:"
            << std::endl;
  AddUnstridedManaged add_unstrided_managed(max_num_blocks, max_block_size);
  add_unstrided_managed.Run(4096, 256);

  std::cout << "\n==> Add with stride kernel and without managed memory :"
            << std::endl;
  AddStridedUnmanaged add_strided_unmanaged(max_num_blocks, max_block_size);
  add_strided_unmanaged.Run(4096, 256);

  std::cout << "\n==> Add without stride kernel and without managed memory:"
            << std::endl;
  AddUnstridedUnmanaged add_unstrided_unmanaged(max_num_blocks, max_block_size);
  add_unstrided_unmanaged.Run(4096, 256);

  std::cout << "\n==> Euclidian Distance with stride kernel:" << std::endl;
  EuclidianDistanceStrided dist_strided(max_num_blocks, max_block_size);
  dist_strided.Run(4096, 256);

  std::cout << "\n==> Euclidian Distance without stride kernel:" << std::endl;
  EuclidianDistanceUnstrided dist_unstrided(max_num_blocks, max_block_size);
  dist_unstrided.Run(4096, 256);

  std::cout << "\n==> Matrix Multiply kernel:" << std::endl;
  MatrixMultiply matrix_multiply(max_num_blocks, max_block_size);
  matrix_multiply.Run(8192, 32);

  // Grid searches.
  std::cout << "\n******** Comparison ***************************" << std::endl;
  std::cout << "\n==> Add kernel, strided vs unstrided:" << std::endl;
  Optimizer<AddKernelFunc> AddOptimizer;
  AddOptimizer.AddStrategy("Strided, Managed", RunStridedSearch<AddKernelFunc>,
                           &add_strided_managed);
  AddOptimizer.AddStrategy("Unstrided, Managed",
                           RunUnstridedSearch<AddKernelFunc>,
                           &add_unstrided_managed);
  AddOptimizer.OptimizeAll(hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  std::cout << "\n==> Add kernel, strided, managed vs unmanaged:" << std::endl;
  Optimizer<AddKernelFunc> AddManUnManOptimizer;
  AddManUnManOptimizer.AddStrategy("Strided, Managed",
                                   RunStridedSearch<AddKernelFunc>,
                                   &add_strided_managed);
  AddManUnManOptimizer.AddStrategy("Strided, Unmanaged",
                                   RunStridedSearch<AddKernelFunc>,
                                   &add_strided_unmanaged);
  AddManUnManOptimizer.OptimizeAll(hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  std::cout << "\n==> Add kernel, strided vs unstrided, managed vs unmanaged:"
            << std::endl;
  Optimizer<AddKernelFunc> AddFullOptimizer;
  AddFullOptimizer.AddStrategy("Strided, Managed",
                               RunStridedSearch<AddKernelFunc>,
                               &add_strided_managed);
  AddFullOptimizer.AddStrategy("Strided, Unmanaged",
                               RunStridedSearch<AddKernelFunc>,
                               &add_strided_unmanaged);
  AddFullOptimizer.AddStrategy("Unstrided, Managed",
                               RunUnstridedSearch<AddKernelFunc>,
                               &add_unstrided_managed);
  AddFullOptimizer.AddStrategy("Unstrided, Managed",
                               RunUnstridedSearch<AddKernelFunc>,
                               &add_unstrided_managed);
  AddFullOptimizer.AddStrategy("Unstrided, Unmanaged",
                               RunUnstridedSearch<AddKernelFunc>,
                               &add_unstrided_unmanaged);
  AddFullOptimizer.OptimizeAll(hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  std::cout << "\n==> Euclidian Distance kernel, strided vs unstrided:"
            << std::endl;
  Optimizer<DistKernelFunc> DistOptimizer;
  DistOptimizer.AddStrategy("Strided", RunStridedSearch<DistKernelFunc>,
                            &dist_strided);
  DistOptimizer.AddStrategy("Unstrided", RunUnstridedSearch<DistKernelFunc>,
                            &dist_unstrided);
  DistOptimizer.OptimizeAll(hardware_info);

  std::cout << "\n***********************************************" << std::endl;
  std::cout << "\n==> Matrix Multiply kernel:" << std::endl;
  Optimizer<MatrixMultiplyKernelFunc> MatrixMultiplyOptimizer;
  MatrixMultiplyOptimizer.AddStrategy(
      "Unstrided", RunUnstridedSearch<MatrixMultiplyKernelFunc>,
      &matrix_multiply);
  MatrixMultiplyOptimizer.OptimizeAll(hardware_info);

  return 0;
}
