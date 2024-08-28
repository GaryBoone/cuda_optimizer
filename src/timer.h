#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
              << line << std::endl;
    if (abort)
      exit(code);
  }
}

class CudaEvent {
public:
  CudaEvent() { gpuErrchk(cudaEventCreate(&event)); }
  ~CudaEvent() { gpuErrchk(cudaEventDestroy(event)); }

  void record(cudaStream_t stream = 0) {
    gpuErrchk(cudaEventRecord(event, stream));
  }

  void synchronize() { gpuErrchk(cudaEventSynchronize(event)); }

  // Computes the elapsed time between two events (in milliseconds with a
  // resolution of around 0.5 microseconds), according to:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
  float elapsedTime(const CudaEvent &other) {
    float time;
    gpuErrchk(cudaEventElapsedTime(&time, event, other.event));
    return time;
  }

private:
  cudaEvent_t event;
};

class CudaTimer {
public:
  void start() { startEvent.record(); }

  void stop() {
    stopEvent.record();
    stopEvent.synchronize();
  }

  float elapsedMilliseconds() { return startEvent.elapsedTime(stopEvent); }

private:
  CudaEvent startEvent, stopEvent;
};
