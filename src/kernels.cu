#include "./kernels.h"

// Bandwidth: (2 reads + 1 write) * n * sizeof(float)
__global__ void AddWithoutStride(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] + y[i];
  }
}

__global__ void EuclidianDistanceWithStride(int n, float2 *x, float2 *y,
                                            float *distance) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    float2 dp = subtract(y[i], x[i]);
    float dist = sqrtf(dot(dp, dp));
    distance[i] = dist;
  }
}
