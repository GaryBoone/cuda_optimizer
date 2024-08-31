#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "adaptive_sampler.h"
#include "reporter.h"
#include "timer.h"

// Build:
// $ cmake -B build -S .
// $ cmake --build build
// Run:
// $ ./build/src/cuda_optimizer
// Test:
// $ ./build/tests/test_app

typedef void (*kernelFuncPtr)(int, float *, float *);

// Bandwidth: (2 reads + 1 write) * n * sizeof(float)
__global__ void AddWithStride(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

// Bandwidth: (2 reads + 1 write) * n * sizeof(float)
__global__ void AddWithoutStride(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] + y[i];
  }
}

// Same as AddWithStride.
__global__ void Add3(int n, float *x, float *y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    {
      y[i] = x[i] + y[i];
    }
  }
}

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

float TimeKernel(kernelFuncPtr kFunc, int num_blocks, int block_size, int n,
                 float *x, float *y) {
  CudaTimer timer;
  timer.Start();
  kFunc<<<num_blocks, block_size>>>(n, x, y);
  timer.Stop();

  // Wait for GPU to finish before accessing on host.
  cudaDeviceSynchronize();

  return timer.ElapsedMilliseconds();
}

int HardwareInfo() {
  int max_threads_per_SM = 0;
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);  // Get the number of devices
  if (num_devices == 0) {
    std::cout << "No CUDA devices found." << std::endl;
    return 0;
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
    max_threads_per_SM = props.maxThreadsPerMultiProcessor;
    std::cout << "  Amount of shared memory per SM: "
              << props.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "  Number of registers per SM: " << props.regsPerMultiprocessor
              << std::endl;
  }
  return max_threads_per_SM;
}

tl::expected<AdaptiveSampler, ErrorInfo> RepeatUntil(double goal_rp,
                                                     kernelFuncPtr kernel_fn,
                                                     int num_blocks,
                                                     int block_size, int n,
                                                     float *x, float *y) {
  AdaptiveSampler stats(goal_rp);
  bool skip_first = true;
  while (stats.ShouldContinue()) {
    // Initialize x and y arrays on the host.
    for (int j = 0; j < n; j++) {
      x[j] = 1.0f;
      y[j] = 2.0f;
    }

    float time = TimeKernel(kernel_fn, num_blocks, block_size, n, x, y);

    if (0 != CheckResult(y, n)) {
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

void RunStrideVarations(int max_threads_per_SM) {
  int N = 1 << 20;  // Run kernel on 1M elements on the GPU.
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU.
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // std::cout << "Running variation with " << numBlocks << " blocks of size "
  //           << blockSize << " and stride " << stride << std::endl;
  // RepeatUntil(0.3, AddWithStride, numBlocks, blockSize, N, x, y);

  float time_at_max_bandwidth = 1e20;
  float max_bandwidth = 0.0;
  int max_bw_block_size = 0;
  int max_bw_num_blocks = 0;

  for (int blockSize = 32; blockSize <= max_threads_per_SM; blockSize += 32) {
    for (int f = 1; f <= (2 << 19); f *= 2) {
      int numBlocks = (N + f - 1) / f;
      // int numBlocks = (N + f * blockSize - 1) / (f * blockSize);
      Reporter::PrintResultsHeader(numBlocks, blockSize);
      auto stats_res =
          RepeatUntil(0.3, AddWithStride, numBlocks, blockSize, N, x, y);

      if (!stats_res) {
        std::cout << " [failed]" << std::endl;
        continue;
      }
      auto mean_res = stats_res->EstimatedMean();
      if (!mean_res || 0.0 == *mean_res) {
        std::cout << " [failed, mean==0.0!]" << std::endl;
        continue;
      }
      auto time_in_seconds = *mean_res / 1000.0;
      auto bandwidth = 3 * N * sizeof(float) / time_in_seconds;
      Reporter::PrintResultsData(bandwidth, *mean_res, stats_res->NumSamples());
      if (bandwidth > max_bandwidth) {
        time_at_max_bandwidth = *mean_res;
        max_bandwidth = bandwidth;
        max_bw_num_blocks = numBlocks;
        max_bw_block_size = blockSize;
      }
    }
    Reporter::PrintResults("current best: ", max_bw_num_blocks,
                           max_bw_block_size, max_bandwidth,
                           time_at_max_bandwidth);
  }

  Reporter::PrintResults("==> final best: ", max_bw_num_blocks,
                         max_bw_block_size, max_bandwidth,
                         time_at_max_bandwidth);

  // Free memory.
  cudaFree(x);
  cudaFree(y);
}

int main(void) {
  auto max_threads_per_SM = HardwareInfo();

  RunStrideVarations(max_threads_per_SM);

  return 0;
}
