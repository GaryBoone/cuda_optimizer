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
__global__ void add_with_stride(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

// Bandwidth: (2 reads + 1 write) * n * sizeof(float)
__global__ void add_no_stride(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] + y[i];
}

// Same as add.
__global__ void add3(int n, float *x, float *y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = x[i] + y[i];
  }
}

void check_result(float *y,
                  int N) { // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  if (maxError > 0.0) {
    std::cout << "  max error: " << maxError << std::endl;
  }
}

float timeKernel(kernelFuncPtr kFunc, int numBlocks, int blockSize, int N,
                 float *x, float *y) {
  CudaTimer timer;
  timer.start();
  kFunc<<<numBlocks, blockSize>>>(N, x, y);
  timer.stop();

  // Wait for GPU to finish before accessing on host.
  cudaDeviceSynchronize();

  return timer.elapsedMilliseconds();
}

void hardware_info() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices); // Get the number of devices
  if (numDevices == 0) {
    std::cout << "No CUDA devices found." << std::endl;
    return;
  }

  std::cout << "Number of CUDA devices: " << numDevices << std::endl;
  cudaDeviceProp props;
  for (int i = 0; i < numDevices; i++) {
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

void repeat_until(double goal_ci, kernelFuncPtr kFunc, int numBlocks,
                  int blockSize, int N, float *x, float *y) {
  AdaptiveSampler stats(0.15);
  bool skip_first = true;
  while (stats.should_continue()) {
    // Initialize x and y arrays on the host.
    for (int j = 0; j < N; j++) {
      x[j] = 1.0f;
      y[j] = 2.0f;
    }
    float time = timeKernel(kFunc, numBlocks, blockSize, N, x, y);
    check_result(y, N);
    // Don't include the first run in the averages to ignore loading effects.
    if (skip_first) {
      skip_first = false;
      continue;
    }
    stats.update(time);
  }

  if (auto est = stats.get_estimate()) {
    std::cout << "  elapsed time: " << *est << " ms, avg over "
              << stats.get_num_samples() << " runs" << std::endl;
    if (*est != 0.0) {
      auto time_in_seconds = *est / 1000.0;
      auto bandwidth = 3 * N * sizeof(float) / time_in_seconds;
      std::cout << "  bandwidth: " << formatNumber(bandwidth) << "B/s"
                << std::endl;
    }
  }
}

int main(void) {
  hardware_info();

  int N = 1 << 20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU.
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // Run kernel on 1M elements on the GPU.
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  std::cout << "> add_with_stride(): " << std::endl;
  repeat_until(0.005, add_with_stride, numBlocks, blockSize, N, x, y);
  std::cout << "> add_no_stride(): " << std::endl;
  repeat_until(0.005, add_no_stride, numBlocks, blockSize, N, x, y);
  std::cout << "> add3(): " << std::endl;
  repeat_until(0.005, add3, numBlocks, blockSize, N, x, y);

  // Free memory.
  cudaFree(x);
  cudaFree(y);

  return 0;
}
