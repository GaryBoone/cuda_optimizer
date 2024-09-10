#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./grid_searchers.h"
#include "./metrics.h"
#include "./reporter.h"

template <typename KernelFunc>
class Optimizer {
 public:
  using SearchFunction = Metrics (*)(cudaDeviceProp, IKernel<KernelFunc>&);

  Optimizer() : name_("Multi-search Optimizer") {}

  explicit Optimizer(std::string name) : name_(std::move(name)) {}

  void OptimizeAll(cudaDeviceProp hardware_info) {
    for (auto& search : searches_) {
      std::cout << "\n*********************************************"
                << std::endl;
      std::cout << "Running " << search.name << " optimization..." << std::endl;
      search.result = search.func(hardware_info, *search.kernel);
      PrintCurrentResults("Current ", search.name, search.result);
    }
    PrintBestResults();
  }

  void AddStrategy(const std::string& name, SearchFunction func,
                   IKernel<KernelFunc>* kernel) {
    searches_.push_back(
        {name, func, std::unique_ptr<IKernel<KernelFunc>>(kernel)});
  }

  const std::string& GetName() const { return name_; }

 private:
  struct Search {
    std::string name;
    SearchFunction func;
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
    std::cout << "\n*********************************************" << std::endl;
    std::cout << "******** Results ******************************" << std::endl;
    std::cout << "Among the following kernels: " << std::endl;
    for (auto& search : searches_) {
      std::cout << "    " << search.name << std::endl;
    }
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
        searches_.begin(), searches_.end(),
        [condition](const Search& a, const Search& b) {
          return !a.result.IsBetter(a.result.get_metrics(condition),
                                    b.result.get_metrics(condition), condition);
        });

    if (it != searches_.end()) {
      std::cout << label << " achieved by " << it->name
                << " kernel:" << std::endl;
      Reporter::PrintResults("  ", it->result.get_metrics(condition));
    }
  }

  std::string name_;
  std::vector<Search> searches_;
};
