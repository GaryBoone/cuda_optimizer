#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#include "adaptive_sampler.h"
#include "format_number.h"
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

void CheckResult(float *y, int N) {
  float max_error = 0.0f;
  for (int i = 0; i < N; i++) {
    max_error = fmax(max_error, fabs(y[i] - 3.0f));
  }
  if (max_error > 0.0) {
    std::cout << "  max error: " << max_error << std::endl;
  }
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

void HardwareInfo() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices); // Get the number of devices
  if (num_devices == 0) {
    std::cout << "No CUDA devices found." << std::endl;
    return;
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
    std::cout << "  Amount of shared memory per SM: "
              << props.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "  Number of registers per SM: " << props.regsPerMultiprocessor
              << std::endl;
  }
}

void RepeatUntil(double goal_rp, kernelFuncPtr kernel_fn, int num_blocks,
                 int block_size, int n, float *x, float *y) {
  AdaptiveSampler stats(goal_rp);
  bool skip_first = true;
  while (stats.ShouldContinue()) {

    // Initialize x and y arrays on the host.
    for (int j = 0; j < n; j++) {
      x[j] = 1.0f;
      y[j] = 2.0f;
    }

    float time = TimeKernel(kernel_fn, num_blocks, block_size, n, x, y);

    CheckResult(y, n);
    // Don't include the first run in the averages to ignore loading effects.
    if (skip_first) {
      skip_first = false;
      continue;
    }
    stats.Update(time);
  }

  if (auto est = stats.EstimatedMean()) {
    std::cout << "  elapsed time: " << *est << " ms, avg over "
              << stats.NumSamples() << " runs" << std::endl;
    if (*est != 0.0) {
      auto time_in_seconds = *est / 1000.0;
      auto bandwidth = 3 * n * sizeof(float) / time_in_seconds;
      std::cout << "  bandwidth: " << FormatNumber(bandwidth) << "B/s"
                << std::endl;
    }
  }
}

int main(void) {
  HardwareInfo();

  int N = 1 << 20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU.
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // Run kernel on 1M elements on the GPU.
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  std::cout << "> add_with_stride(): " << std::endl;
  RepeatUntil(0.15, AddWithStride, numBlocks, blockSize, N, x, y);
  std::cout << "> add_no_stride(): " << std::endl;
  RepeatUntil(0.15, AddWithoutStride, numBlocks, blockSize, N, x, y);
  std::cout << "> add3(): " << std::endl;
  RepeatUntil(0.15, Add3, numBlocks, blockSize, N, x, y);

  // Free memory.
  cudaFree(x);
  cudaFree(y);

  return 0;
}
