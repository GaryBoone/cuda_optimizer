#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./grid_searchers.h"
#include "./metrics.h"
#include "./reporter.h"

namespace cuda_optimizer {

template <typename KernelFunc>
class Optimizer {
 public:
  using SearchFunction = Metrics (*)(cudaDeviceProp, IKernel<KernelFunc>&);

  Optimizer() : name_("Multi-search Optimizer") {}

  explicit Optimizer(std::string name) : name_(std::move(name)) {}

  void AddStrategy(const std::string& name, SearchFunction func,
                   IKernel<KernelFunc>* kernel) {
    searches_[name] = {name, func,
                       std::unique_ptr<IKernel<KernelFunc>>(kernel)};
  }
  void CreateSet(const std::string& set_name,
                 const std::vector<std::string>& strategy_names) {
    sets_[set_name] = strategy_names;
  }

  const std::string& GetName() const { return name_; }

  void OptimizeSet(const std::string& set_name, cudaDeviceProp hardware_info) {
    std::cout << "Running optimization set: " << set_name << std::endl;

    const auto& strategy_names = sets_[set_name];
    for (const auto& name : strategy_names) {
      auto& search = searches_[name];
      if (!search.result.has_value()) {
        std::cout << "Running " << name << " optimization..." << std::endl;
        search.result = search.func(hardware_info, *search.kernel);
      } else {
        std::cout << "Using cached results for " << name << std::endl;
      }
      PrintCurrentResults("Current ", name, *search.result);
    }
    PrintBestResults(set_name);
  }

 private:
  struct Strategy {
    SearchFunction func;
    std::unique_ptr<IKernel<KernelFunc>> kernel;
  };

  struct Search {
    std::string name;
    SearchFunction func;
    std::unique_ptr<IKernel<KernelFunc>> kernel;
    std::optional<Metrics> result;
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

  void PrintBestResults(const std::string& set_name) const {
    std::cout << "\n*********************************************" << std::endl;
    std::cout << "*** Results for set: " << set_name << " *******" << std::endl;
    std::cout << "Among the following kernels: " << std::endl;
    for (const auto& name : sets_.at(set_name)) {
      std::cout << "    " << name << std::endl;
    }
    PrintBestResult(set_name, "Best time", Condition::kMinTime,
                    [](const Metrics& m) {
                      return m.get_metrics(Condition::kMinTime).time_ms;
                    });
    PrintBestResult(set_name, "Best bandwidth", Condition::kMaxBandwidth,
                    [](const Metrics& m) {
                      return m.get_metrics(Condition::kMaxBandwidth).bandwidth;
                    });
    PrintBestResult(set_name, "Best occupancy", Condition::kMaxOccupancy,
                    [](const Metrics& m) {
                      return m.get_metrics(Condition::kMaxOccupancy).occupancy;
                    });
  }

  template <typename Getter>
  void PrintBestResult(const std::string& set_name, const std::string& label,
                       Condition condition, Getter getter) const {
    const auto& strategy_names = sets_.at(set_name);
    auto it = std::max_element(
        strategy_names.begin(), strategy_names.end(),
        [this, condition](const std::string& a, const std::string& b) {
          const auto& result_a = searches_.at(a).result.value();
          const auto& result_b = searches_.at(b).result.value();
          return !result_a.IsBetter(result_a.get_metrics(condition),
                                    result_b.get_metrics(condition), condition);
        });

    if (it != strategy_names.end()) {
      std::cout << label << " achieved by " << *it << " kernel:" << std::endl;
      Reporter::PrintResults(
          "  ", searches_.at(*it).result.value().get_metrics(condition));
    }
  }

  std::string name_;
  std::map<std::string, Search> searches_;
  std::map<std::string, std::vector<std::string>> sets_;
};

}  // namespace cuda_optimizer
