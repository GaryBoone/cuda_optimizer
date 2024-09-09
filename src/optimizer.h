#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>

class OptimizerBase {
 public:
  virtual ~OptimizerBase() = default;
  virtual void Optimize(cudaDeviceProp hardware_info, void* kernel) const = 0;
  const std::string& GetName() const { return name_; }

 protected:
  explicit OptimizerBase(std::string name) : name_(std::move(name)) {}

 private:
  std::string name_;
};

template <typename KernelFunc>
class Optimizer : public OptimizerBase {
 public:
  using VariationFunction = void (*)(cudaDeviceProp, IKernel<KernelFunc>&);

  Optimizer(std::string name, VariationFunction func)
      : OptimizerBase(std::move(name)), variation_func_(func) {}

  void Optimize(cudaDeviceProp hardware_info, void* kernel) const override {
    std::cout << "Running " << GetName() << " optimization..." << std::endl;
    variation_func_(hardware_info, *static_cast<IKernel<KernelFunc>*>(kernel));
  }

  // static std::unique_ptr<OptimizerBase> CreateStrided() {
  //   return std::make_unique<Optimizer<KernelFunc>>(
  //       "Strided", RunStrideVariations<KernelFunc>);
  // }

  // static std::unique_ptr<OptimizerBase> CreateUnstrided() {
  //   return std::make_unique<Optimizer<KernelFunc>>(
  //       "Unstrided", RunUnstridedVariations<KernelFunc>);
  // }

 private:
  VariationFunction variation_func_;
};

// Helper function to create an optimizer
template <typename KernelFunc>
std::unique_ptr<OptimizerBase> CreateOptimizer(
    const std::string& name,
    void (*func)(cudaDeviceProp, IKernel<KernelFunc>&)) {
  return std::make_unique<Optimizer<KernelFunc>>(name, func);
}
