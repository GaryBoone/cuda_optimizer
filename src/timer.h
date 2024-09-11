#pragma once

#include <cuda_runtime.h>

#include <iostream>

namespace cuda_optimizer {

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "timer.h GPUassert: " << cudaGetErrorString(code)
              << " in file '" << file << "', line " << line << std::endl;
    if (abort) exit(code);
  }
}

class CudaEvent {
 public:
  CudaEvent() { gpuErrchk(cudaEventCreate(&event_)); }
  ~CudaEvent() { gpuErrchk(cudaEventDestroy(event_)); }

  void Record(cudaStream_t stream = 0) {
    gpuErrchk(cudaEventRecord(event_, stream));
  }

  void Synchronize() { gpuErrchk(cudaEventSynchronize(event_)); }

  // Computes the elapsed time between two events (in milliseconds with a
  // resolution of around 0.5 microseconds), according to:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
  float ElapsedTime(const CudaEvent &other) {
    float time;
    gpuErrchk(cudaEventElapsedTime(&time, event_, other.event_));
    return time;
  }

 private:
  cudaEvent_t event_;
};

class CudaTimer {
 public:
  void Start() { start_event_.Record(); }

  void Stop() {
    stop_event_.Record();
    stop_event_.Synchronize();
  }

  float ElapsedMilliseconds() { return start_event_.ElapsedTime(stop_event_); }

 private:
  CudaEvent start_event_, stop_event_;
};

}  // namespace cuda_optimizer
