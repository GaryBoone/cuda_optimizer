#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

__global__ void AddWithoutStride(int n, float *x, float *y);
__global__ void Add3(int n, float *x, float *y);
__global__ void EuclidianDistanceWithStride(int n, float2 *x, float2 *y,
                                            float *distance);

// Utility functions.
inline __device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
