#include "kernels.h"

namespace cuda_optimizer {

__global__ void AddUnstridedKernel(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] + y[i];
  }
}

__global__ void AddStridedKernel(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

__global__ void EuclideanDistanceUnstridedKernel(int n, float2 *x, float2 *y,
                                                 float *distance) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float2 dp = subtract(y[i], x[i]);
    float dist = sqrtf(dot(dp, dp));
    distance[i] = dist;
  }
}

__global__ void EuclideanDistanceStridedKernel(int n, float2 *x, float2 *y,
                                               float *distance) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    float2 dp = subtract(y[i], x[i]);
    float dist = sqrtf(dot(dp, dp));
    distance[i] = dist;
  }
}

__global__ void MatrixMultiplyKernel(int matrix_dim, float *a, float *b,
                                     float *c) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = thread_id / matrix_dim;
  int col = thread_id % matrix_dim;

  if (row < matrix_dim && col < matrix_dim) {
    float sum = 0.0f;
    for (int k = 0; k < matrix_dim; k++) {
      sum += a[row * matrix_dim + k] * b[k * matrix_dim + col];
    }
    c[row * matrix_dim + col] = sum;
  }
}

}  // namespace cuda_optimizer
