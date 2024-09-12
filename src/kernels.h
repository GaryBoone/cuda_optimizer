#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace cuda_optimizer {

__global__ void AddStridedKernel(int n, float *x, float *y);
__global__ void AddUnstridedKernel(int n, float *x, float *y);

__global__ void EuclideanDistanceStridedKernel(int n, float2 *x, float2 *y,
                                               float *distance);
__global__ void EuclideanDistanceUnstridedKernel(int n, float2 *x, float2 *y,
                                                 float *distance);

__global__ void MatrixMultiplyKernel(int matrix_dim, float *a, float *b,
                                     float *c);

// The Add kernels are defined like:
//     __global__ void AddStridedKernel(int n, float *x, float *y);
// so here is their shared type.
using AddKernelFunc = void (*)(int, float *, float *);

// The Euclidean Distance kernels are defined like:
//     __global__ void EuclideanDistanceStridedKernel(int n, float2 *x,
//                                                    float2 *y,
//                                                    float *distance);
// so here is their shared type.
using DistKernelFunc = void (*)(int, float2 *, float2 *, float *);

// The Matrix Multiply kernels are defined like:
//    __global__ void MatrixMultiplyKernel(int N, float *A, float *B, float *C);
// so here is their shared type.
using MatrixMultiplyKernelFunc = void (*)(int, float *, float *, float *);

// Matrices, used in the the matrix multiply kernel, are stored in row-major
// order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  float *elements;
} Matrix;

// Utility functions.
inline __device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

}  // namespace cuda_optimizer
