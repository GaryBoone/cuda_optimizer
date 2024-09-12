#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "examples/add_strided_managed.h"
#include "examples/add_strided_unmanaged.h"
#include "examples/add_unstrided_managed.h"
#include "examples/add_unstrided_unmanaged.h"
#include "examples/euclidean_distance_strided.h"
#include "examples/euclidean_distance_unstrided.h"
#include "examples/matrix_multiply.h"
#include "optimizer.h"

namespace co = ::cuda_optimizer;

cudaDeviceProp HardwareInfo() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);  // Get the number of devices
  if (num_devices == 0) {
    std::cout << "No CUDA devices found. Exitting." << std::endl;
    exit(1);
  }

  std::cout << "Number of CUDA devices: " << num_devices << std::endl;
  cudaDeviceProp props;
  // If there are multiple devices, print all info and return the last one.
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

int main(void) {
  auto hardware_info = HardwareInfo();
  int max_num_blocks = hardware_info.maxThreadsDim[0] *
                       hardware_info.maxThreadsDim[1] *
                       hardware_info.maxThreadsDim[2];
  int max_block_size = hardware_info.maxThreadsPerBlock;
  std::cout << "...which means that: " << std::endl;
  std::cout << "  max_num_blocks: "
            << co::Reporter::FormatWithCommas(max_num_blocks) << std::endl;
  std::cout << "  max_block_size: "
            << co::Reporter::FormatWithCommas(max_block_size) << std::endl;

  // Individual runs.
  std::cout << "\n==> Add with stride kernel and with managed memory :"
            << std::endl;
  co::AddStridedManaged add_strided_managed(max_num_blocks, max_block_size);
  add_strided_managed.Run(4096, 256);

  std::cout << "\n==> Add without stride kernel and with managed memory:"
            << std::endl;
  co::AddUnstridedManaged add_unstrided_managed(max_num_blocks, max_block_size);
  add_unstrided_managed.Run(4096, 256);

  std::cout << "\n==> Add with stride kernel and without managed memory :"
            << std::endl;
  co::AddStridedUnmanaged add_strided_unmanaged(max_num_blocks, max_block_size);
  add_strided_unmanaged.Run(4096, 256);

  std::cout << "\n==> Add without stride kernel and without managed memory:"
            << std::endl;
  co::AddUnstridedUnmanaged add_unstrided_unmanaged(max_num_blocks,
                                                    max_block_size);
  add_unstrided_unmanaged.Run(4096, 256);

  std::cout << "\n==> Euclidean Distance with stride kernel:" << std::endl;
  co::EuclideanDistanceStrided dist_strided(max_num_blocks, max_block_size);
  dist_strided.Run(4096, 256);

  std::cout << "\n==> Euclidean Distance without stride kernel:" << std::endl;
  co::EuclideanDistanceUnstrided dist_unstrided(max_num_blocks, max_block_size);
  dist_unstrided.Run(4096, 256);

  std::cout << "\n==> Matrix Multiply kernel:" << std::endl;
  co::MatrixMultiply matrix_multiply(max_num_blocks, max_block_size);
  matrix_multiply.Run(8192, 32);

  // Grid searches.
  co::Optimizer<co::AddKernelFunc> optimizer;
  optimizer.AddStrategy("Strided, Managed",
                        co::RunStridedSearch<co::AddKernelFunc>,
                        &add_strided_managed);
  optimizer.AddStrategy("Unstrided, Managed",
                        co::RunUnstridedSearch<co::AddKernelFunc>,
                        &add_unstrided_managed);
  optimizer.AddStrategy("Strided, Unmanaged",
                        co::RunStridedSearch<co::AddKernelFunc>,
                        &add_strided_unmanaged);
  optimizer.AddStrategy("Unstrided, Unmanaged",
                        co::RunUnstridedSearch<co::AddKernelFunc>,
                        &add_unstrided_unmanaged);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  auto name = "Add kernel, strided vs unstrided [both managed]";
  optimizer.CreateSet(name, {"Strided, Managed", "Unstrided, Managed"});
  optimizer.OptimizeSet(name, hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  name = "Add kernel, managed vs unmanaged [both strided]";
  optimizer.CreateSet(name, {"Strided, Managed", "Strided, Unmanaged"});
  optimizer.OptimizeSet(name, hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  name = "Add kernel, strided/unstrided and managed/unmanaged";
  optimizer.CreateSet(name, {"Strided, Managed", "Unstrided, Managed",
                             "Strided, Unmanaged", "Unstrided, Unmanaged"});
  optimizer.OptimizeSet(name, hardware_info);

  std::cout << "\n******** Comparison ***************************" << std::endl;
  co::Optimizer<co::DistKernelFunc> DistOptimizer;
  DistOptimizer.AddStrategy("Strided", co::RunStridedSearch<co::DistKernelFunc>,
                            &dist_strided);
  DistOptimizer.AddStrategy(
      "Unstrided", co::RunUnstridedSearch<co::DistKernelFunc>, &dist_unstrided);
  name = "Euclidean Distance kernel, strided vs unstrided";
  optimizer.CreateSet(name, {"Strided", "Unstrided"});
  DistOptimizer.OptimizeSet(name, hardware_info);

  std::cout << "\n***********************************************" << std::endl;
  co::Optimizer<co::MatrixMultiplyKernelFunc> matrix_multiply_optimizer;
  matrix_multiply_optimizer.AddStrategy(
      "Unstrided", co::RunUnstridedSearch<co::MatrixMultiplyKernelFunc>,
      &matrix_multiply);
  name = "Matrix Multiply kernel";
  optimizer.CreateSet(name, {"Unstrided"});
  matrix_multiply_optimizer.OptimizeSet(name, hardware_info);

  return 0;
}