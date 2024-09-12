# CUDA Optimizer

*Optimizes CUDA kernels by searching for the best parameters and optimization
methods.*

GPUs provide incredible speedups for AI and other numerically intensive
tasks, but require deep knowledge to use effectively. While writing kernels is
straightforward, how they interact with GPU architectures can have dramatic
effects on speed and throughput.

There are multiple ways to structure kernels and incorporate GPU architecture
knowledge to optimize them, such as *striding*, *occupancy*, *coalesced memory
access*, *shared memory*, *thread block size optimization*, *register pressure
management*, and more.

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
  2. managed vs unmanaged memory,
  3. strided vs unstrided loops,
  4. occupancy
* Allows optimizations to be groups into sets for multi-way optimization.
* Statistically accurate and flexible timer for measuring kernel timing.

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
$ ./build/src/cuda_optimizer | tee /tmp/cuda_optimizer_out.txt
$ less /tmp/cuda_optimizer_out.txt
```

## Optimization

Groups of kernels can be created in the Optimizer and run as a set to make
comparisons. For example, to compare strided vs unstrided variations of a
Euclidean Distance kernel, we can do:

```c++
  // Make an optimizer for the DistKernelFunc.
  Optimizer<DistKernelFunc> DistOptimizer;

  // Add strategies.
  DistOptimizer.AddStrategy("Strided",      RunStridedSearch<DistKernelFunc>,
                            &dist_strided);
  DistOptimizer.AddStrategy(
      "Unstrided", RunUnstridedSearch<DistKernelFunc>, &dist_unstrided);

  // Create comparison set with these two strategies.
  name = "Euclidean Distance kernel, strided vs unstrided";
  optimizer.CreateSet(name, {"Strided", "Unstrided"});

  // Optimize and compare the two strategies.
  DistOptimizer.OptimizeSet(name, hardware_info);
```

## Understanding the output

As it runs, `cuda_optimizer` outputs lines like:

```text
<<numBlocks, blockSize>> = <<     1,024,   352>>, occupancy:  0.92, bandwidth:      8.32GB/s, time:    1.51 ms (over 34 runs)
```

This line shows a parameter variation over `numBlocks` and `blockSize` in which
the kernel, in this case a vector add with striding, is being called like:

```text
AddStrided<<1024, 352>>(n, x, y);
```

The line shows that the result had an occupancy of 92%, bandwidth of 8.32GB/s,
and an average run time of 1.51 ms. The problem size is 1<<20, or 1,048,576,
specified in the `add_strided_managed.h` file.

At the end of a run, the final results are printed:

```text
 Results for set: Add kernel, strided vs unstrided [both managed] *******
Among the following kernels:
    Strided, Managed
    Unstrided, Managed
Best time achieved by Unstrided, Managed kernel:
  <<numBlocks, blockSize>> = <<     4,096,   256>>
                  occupancy:  1.00, bandwidth:     17.88GB/s, time:    0.70 ms
Best bandwidth achieved by Unstrided, Managed kernel:
  <<numBlocks, blockSize>> = <<     4,096,   256>>
                  occupancy:  1.00, bandwidth:     17.88GB/s, time:    0.70 ms
Best occupancy achieved by Unstrided, Managed kernel:
  <<numBlocks, blockSize>> = <<     4,096,   256>>
                  occupancy:  1.00, bandwidth:     17.88GB/s, time:    0.70 ms

```

Here we see that, counter to expectations, striding was not the optimal loop
style. Instead the Unstrided version was fastest and the highest bandwidth in
the case where both used Cuda's managed memory feature. Unsurprisingly, the best
results were achieved with 100% occupancy.

## Timing

Each run is repeated over sufficient runs achieve a given level of statistical
accuracy. Specifically, it continues sampling until the margin of error (at 95%
confidence) is less than a specified fraction (`relative_precision_`) of the
sample standard deviation.

For example, in a normal curve, ~0.68% of the population is found within +/- 1
standard deviation of the mean. The timer used in `cuda_optimizer` allows the
developer to set this precision value to balance accuracy with test length. In
practice, a required precision of 0.35 is satisfied with about 34 runs.

## Architecture

The code is organized into a series of layers:

* Optimizer: collects examples into sets that can be run and compared. For
  example, we can compared managed to unmanaged, strided to unstrided, or all
  together.

* Examples: Implement the `IKernel` interface, templated on the kernel.

```c++
class AddStridedManaged : public IKernel<AddKernelFunc> {
```

The `IKernel` interface provides `Setup()`, `RunKernel()`, `Cleanup()`, and `CheckResults()`. Examples also include the generators use for grid searching over their parameters.

* kernels: These are the base Cuda kernels.

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

namespace cuda_optimizer {

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

} // namespace cuda_optimizer
```

The kernel has been moved to `kernels.cu`/`kernels.h`. Note that `numBlocks` and
`blockSize` become inputs determined by the call to `RunKernel()` or by the
optimizers.

## Readability

The project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Note that the CUDA function calls start with lowercase letters, while Google style is to start function calls with Capital letters. This difference makes it easy to distinguish the CUDA calls.

## TODO

* Add an output comparison matrix. It would show the relative speed ratio
  between any two strategies in an n-way comparison. So for example, it would
  show that `Strided & Managed` vs `Unstrided and Unmanaged` has a bandwidth
  ratio of 1.2, meaning that `Strided & Managed` delivered 20% more bandwidth.

## License

Distributed under the MIT License. See `LICENSE-MIT.md` for more information.
