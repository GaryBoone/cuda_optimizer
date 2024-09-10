#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./metrics.h"
#include "./reporter.h"

template <typename KernelFunc>
class Optimizer {
 public:
  using VariationFunction = Metrics (*)(cudaDeviceProp, IKernel<KernelFunc>&);

  Optimizer() : name_("Multi-strategy Optimizer") {}

  explicit Optimizer(std::string name) : name_(std::move(name)) {}

  void OptimizeAll(cudaDeviceProp hardware_info) {
    for (auto& strategy : strategies_) {
      std::cout << "\n*******************************************" << std::endl;
      std::cout << "Running " << strategy.name << " optimization..."
                << std::endl;
      strategy.result = strategy.func(hardware_info, *strategy.kernel);
      PrintCurrentResults("Current ", strategy.name, strategy.result);
    }
    PrintBestResults();
  }

  void AddStrategy(const std::string& name, VariationFunction func,
                   IKernel<KernelFunc>* kernel) {
    strategies_.push_back(
        {name, func, std::unique_ptr<IKernel<KernelFunc>>(kernel)});
  }

  const std::string& GetName() const { return name_; }

 private:
  struct Strategy {
    std::string name;
    VariationFunction func;
    std::unique_ptr<IKernel<KernelFunc>> kernel;
    Metrics result;
  };

  void PrintCurrentResults(std::string header, std::string name,
                           Metrics result) {
    Reporter::PrintResults(header + name + " best      time: ",
                           result.get_metrics(Condition::kMinTime));
    Reporter::PrintResults(header + name + " best  bandwith: ",
                           result.get_metrics(Condition::kMaxBandwidth));
    Reporter::PrintResults(header + name + " best occupancy: ",
                           result.get_metrics(Condition::kMaxOccupancy));
  }

  void PrintBestResults() const {
    std::cout << "\n================  Results ===================" << std::endl;
    PrintBestResult("Best time", Condition::kMinTime, [](const Metrics& m) {
      return m.get_metrics(Condition::kMinTime).time_ms;
    });
    PrintBestResult("Best bandwidth", Condition::kMaxBandwidth,
                    [](const Metrics& m) {
                      return m.get_metrics(Condition::kMaxBandwidth).bandwidth;
                    });
    PrintBestResult("Best occupancy", Condition::kMaxOccupancy,
                    [](const Metrics& m) {
                      return m.get_metrics(Condition::kMaxOccupancy).occupancy;
                    });
  }

  template <typename Getter>
  void PrintBestResult(const std::string& label, Condition condition,
                       Getter getter) const {
    auto it = std::max_element(
        strategies_.begin(), strategies_.end(),
        [condition](const Strategy& a, const Strategy& b) {
          return !a.result.IsBetter(a.result.get_metrics(condition),
                                    b.result.get_metrics(condition), condition);
        });

    if (it != strategies_.end()) {
      std::cout << label << " achieved by " << it->name
                << " kernel:" << std::endl;
      Reporter::PrintResults("  ", it->result.get_metrics(condition));
    }
  }

  std::string name_;
  std::vector<Strategy> strategies_;
};
