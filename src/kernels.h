#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

// Utility functions.
inline __device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
