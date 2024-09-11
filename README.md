# CUDA Optimizer

_Optimizes CUDA kernels by searching for the best parameters and optimization
methods._

GPUs provide incredible speedups for AI and other numerically intensive
tasks, but require deep knowledge to use effectively. While writing kernels is
straightforward, how they interact with GPU architectures can have dramatic
effects on speed and throughput.

There are multiple ways to structure kernels and incorporate GPU architecture
knowledge to optimize them, such as _striding_, _occupancy_, _coalesced memory
access_, _shared memory_, _thread block size optimization_, _register pressure
management_, and more.

The problem is that a developer can't know in advance how these optimizations
interact, or which combinations are most effective, or which actually interfere
with others.

This repository provides code that compares optimization techniques for common
kernels and provides a framework for optimizing your kernels.

## Features
* Finds optimal parameters for speed and throughput of CUDA kernels.
* Allows kernels to be included in multiple optimizations.
* Includes common metrics like time, bandwidth, and occupancy.
* Provides generators for common grid searches and architecture-appropriate
  values for kernels. For example, it includes predefined searching by
  warp-sized increments.
* Optimizes over:
  1. `numBlocks` and `blockSize` kernel parameters.
  1. managed vs unmanaged memory,
  1. strided vs unstrided loops,
  1. occupancy
* Allows optimizations to be groups into sets for multi-way optimization.

## How to build and run

The code is `CMake` based, so use the standard procedure:

Build:
```bash
  # cd into the project directory
  $ mkdir build
  $ cmake -B build -S .
  $ cmake --build build
```

Test:
```bash
  $ ./build/tests/test_app
```

Run:
```bash
  $ ./build/src/cuda_optimizer
```

The output is long, so it's useful to capture the output for review:
```bash
$ ./build/src/cuda_optimizer | tee /tmp/opt_out.txt
$ less /tmp/opt_out.txt
```

## Understanding the output

## Architecture

## Converting an example into the framework

Suppose we have a simple CUDA example that shows a vector add using striding and
managed memory:

```c++
__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU.
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host.
  cudaDeviceSynchronize();

  // Check for errors. All values should be 3.0.
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
```

To convert it to work in the `CUDA Optimizer` framework, we just need to break
this code up into `Setup()`, `RunKernel()`, `Cleanup()`, and `CheckResults()`.
That is, we define a subclass of `IKernel` and break up the code into the
`IKernel` methods. Most of the `.h` file is boilerplate (See
`add_strided_managed.h`). Here's the complete `add_strided_managed.cu` file:

```c++
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "add_strided_managed.h"

void AddStridedManaged::Setup() {
  cudaMallocManaged(&x_, n_ * sizeof(float));
  cudaMallocManaged(&y_, n_ * sizeof(float));

  for (int j = 0; j < n_; j++) {
    x_[j] = 1.0f;
    y_[j] = 2.0f;
  }
}

void AddStridedManaged::RunKernel(int num_blocks, int block_size) {
  AddStridedKernel<<<num_blocks, block_size>>>(n_, x_, y_);
  cudaDeviceSynchronize();
}

void AddStridedManaged::Cleanup() {
  cudaFree(x_);
  cudaFree(y_);
}

int AddStridedManaged::CheckResults() {
  int num_errors = 0;
  double max_error = 0.0;

  for (int i = 0; i < n_; i++) {
    if (fabs(y_[i] - 3.0f) > 1e-6) {
      num_errors++;
    }
    max_error = fmax(max_error, fabs(y_[i] - 3.0f));
  }

  if (num_errors > 0) {
    std::cout << "  number of errors: " << num_errors;
  }
  if (max_error > 0.0) {
    std::cout << ",  max error: " << max_error;
  }
  return num_errors;
}
```
The kernel has been moved to `kernels.cu`/`kernels.h`. Note that `numBlocks` and
`blockSize` become inputs determined by the call to `RunKernel()` or by the
optimizers.

## Readability
The project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Note that the CUDA function calls start with lowercase letters, while Google style is to start function calls with Capital letters. This difference makes it easy to distinguish the CUDA calls.

## License
Distributed under the MIT License. See `LICENSE-MIT.md` for more information.