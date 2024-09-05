#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "adaptive_sampler.h"
#include "example.h"
#include "examples/add.h"
#include "examples/euclidian_distance.h"
#include "kernels.h"
#include "metrics.h"
#include "reporter.h"
#include "timer.h"

// Build:
// $ cmake -B build -S .
// $ cmake --build build
// Run:
// $ ./build/src/cuda_optimizer
// Test:
// $ ./build/tests/test_app

const double kRequiredPrecision = 0.35;

int CheckResult(float *y, int n) {
  int num_errors = 0;
  double max_error = 0.0;

  for (int i = 0; i < n; i++) {
    if (fabs(y[i] - 3.0f) > 1e-6) {
      num_errors++;
    }
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }

  if (num_errors > 0) {
    std::cout << "  number of errors: " << num_errors;
  }
  if (max_error > 0.0) {
    std::cout << ",  max error: " << max_error;
  }
  return num_errors;
}

float TimeKernel(IKernel &ex, int num_blocks, int block_size) {
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

double Occupancy(cudaDeviceProp props, int num_blocks, int block_size,
                 kernelFuncPtr kernel) {
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel, block_size,
                                                0);
  int activeWarps = num_blocks * block_size / props.warpSize;
  assert(0 != props.warpSize);
  int maxWarps = props.maxThreadsPerMultiProcessor / props.warpSize;
  return (static_cast<double>(activeWarps) / maxWarps);
}

tl::expected<AdaptiveSampler, ErrorInfo> RepeatUntil(double required_precision,
                                                     IKernel &ex,
                                                     int num_blocks,
                                                     int block_size) {
  AdaptiveSampler stats(required_precision);
  bool skip_first = true;
  while (stats.ShouldContinue()) {
    ex.Reset();

    float time = TimeKernel(ex, num_blocks, block_size);

    if (0 != ex.CheckResults()) {
      return tl::make_unexpected(ErrorInfo(ErrorInfo::kUnexpectedKernelResult,
                                           "errors in kernel results"));
    }

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
void OptimizeOccupancy(cudaDeviceProp &hardware_info, int &num_blocks,
                       int &block_size, kernelFuncPtr kernel) {
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

void PrintResults(std::string header, Metrics metrics) {
  Reporter::PrintResults(header + " best      time: ",
                         metrics.get_metrics(Condition::kMinTime));
  Reporter::PrintResults(header + " best  bandwith: ",
                         metrics.get_metrics(Condition::kMaxBandwidth));
  Reporter::PrintResults(header + " best occupancy: ",
                         metrics.get_metrics(Condition::kMaxOccupancy));
}

void RunStrideVariations(cudaDeviceProp hardware_info, IKernel &ex) {
  Metrics metrics;

  int numBlocks, blockSize;
  OptimizeOccupancy(hardware_info, numBlocks, blockSize, ex.GetKernel());
  std::cout << "expected optimal num_blocks: " << numBlocks << std::endl;
  std::cout << "expected optimal block_size: " << blockSize << std::endl;

  // kFunc<<<num_blocks, block_size>>>
  //                    block_size <= maxThreadsPerBlock
  //         num_blocks <= maxgridsize
  // kFunc<<<max_num_blocks, max_block_size>>>

  // Allocate Unified Memory – accessible from CPU or GPU.
  ex.Setup();
  auto kernel_info = ex.GetKernelInfo();

  auto block_size_gen = ex.GetBlockSizeGenerator();
  while (auto block_size = block_size_gen->Next()) {
    auto num_blocks_gen = ex.GetNumBlocksGenerator();
    while (auto num_blocks = num_blocks_gen->Next()) {
      if (*num_blocks * *block_size > kernel_info.n) {
        *num_blocks = *num_blocks / 2 * 1.1;  // Try just 10% overprovision.
      }
      Reporter::PrintResultsHeader(*num_blocks, *block_size);
      auto occupancy =
          Occupancy(hardware_info, *num_blocks, *block_size, ex.GetKernel());

      auto stats_res =
          RepeatUntil(kRequiredPrecision, ex, *num_blocks, *block_size);

      if (!stats_res) {
        std::cout << " [failed]" << std::endl;
        continue;
      }
      auto mean_res = stats_res->EstimatedMean();
      if (!mean_res || 0.0 == *mean_res) {
        std::cout << " [failed, mean==0.0!]" << std::endl;
        continue;
      }
      auto time_in_ms = *mean_res;
      auto time_in_seconds = time_in_ms / 1000.0;
      auto bandwidth =
          kernel_info.n * kernel_info.bytesPerElement / time_in_seconds;
      Data current_metrics{*num_blocks, *block_size, time_in_ms, bandwidth,
                           occupancy};
      metrics.UpdateAll(current_metrics);
      Reporter::PrintResultsData(current_metrics, stats_res->NumSamples());

      if (*num_blocks * *block_size > kernel_info.n) {
        // n = 1 << 20 = 1,048,576
        // <<<2097152,  1>>> because 2,097,152 *  1 = 2,097,152 > 1,048,576
        // <<<  32768, 64>>> because    32,768 * 64 = 2,097,152 > 1,048,576
        // Try only one overprovision.
        break;
      }
    }
    PrintResults("current", metrics);
  }
  PrintResults("final", metrics);

  ex.Cleanup();
}

int main(void) {
  auto hardware_info = HardwareInfo();
  int max_num_blocks = hardware_info.maxThreadsDim[0] *
                       hardware_info.maxThreadsDim[1] *
                       hardware_info.maxThreadsDim[2];
  int max_block_size = hardware_info.maxThreadsPerBlock;
  std::cout << "max_num_blocks: " << max_num_blocks << std::endl;
  std::cout << "max_block_size: " << max_block_size << std::endl;

  EuclidianDistance ex;
  ex.run();

  Add add(max_num_blocks, max_block_size);
  RunStrideVariations(hardware_info, add);

  return 0;
}
