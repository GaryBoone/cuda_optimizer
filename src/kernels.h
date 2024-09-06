#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Utility functions.
inline __device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
