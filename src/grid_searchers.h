#pragma once

#include <string>

#include "./metrics.h"
#include "./reporter.h"

// The repeated runs done by the grid searches generate multiple samples for
// how long it takes to run a kernel. The grid searches then use these samples
// to estimate the mean time it takes to run the kernel. The grid searches
// continue to run the kernel until the estimated mean time is within a certain
// precision of the actual mean time. This precision is defined by
// kRequiredPrecision.
const double kRequiredPrecision = 0.35;

inline void PrintResults(std::string header, Metrics metrics) {
  Reporter::PrintResults(header + " best      time: ",
                         metrics.get_metrics(Condition::kMinTime));
  Reporter::PrintResults(header + " best  bandwith: ",
                         metrics.get_metrics(Condition::kMaxBandwidth));
  Reporter::PrintResults(header + " best occupancy: ",
                         metrics.get_metrics(Condition::kMaxOccupancy));
}

template <typename KernelFunc>
Metrics RunStridedSearch(
    cudaDeviceProp hardware_info,
    IKernel<KernelFunc> &ex) {  // NOLINT(runtime/references
  Metrics metrics;

  int num_blocks, block_size;
  OptimizeOccupancy(hardware_info, num_blocks, block_size, ex.GetKernel());
  std::cout << "expected optimal num_blocks: " << num_blocks << std::endl;
  std::cout << "expected optimal block_size: " << block_size << std::endl;

  auto kernel_info = ex.GetKernelInfo();

  auto block_size_gen = ex.GetBlockSizeGenerator();
  while (auto block_size_opt = block_size_gen->Next()) {
    auto num_blocks_gen = ex.GetNumBlocksGenerator();
    while (auto num_blocks_opt = num_blocks_gen->Next()) {
      int num_blocks = *num_blocks_opt;
      int block_size = *block_size_opt;
      if (num_blocks * block_size > kernel_info.n) {
        // Don't double; only overprovision by 10%.
        num_blocks = num_blocks / 2.0 * 1.1;
      }

      Reporter::PrintResultsHeader(num_blocks, block_size);
      auto occupancy =
          Occupancy(hardware_info, num_blocks, block_size, ex.GetKernel());

      auto stats_res =
          RepeatUntil(kRequiredPrecision, ex, num_blocks, block_size);

      if (!stats_res) {
        std::cout << " [failed]" << std::endl;
        continue;
      }
      auto mean_res = stats_res->EstimatedMean();
      if (!mean_res || 0.0 == *mean_res) {
        std::cout << " [failed, mean==0.0!]" << std::endl;
        continue;
      }
      auto time_in_ms = *mean_res;
      auto time_in_seconds = time_in_ms / 1000.0;
      auto bandwidth = kernel_info.total_bytes / time_in_seconds;
      Data current_metrics{num_blocks, block_size, time_in_ms, bandwidth,
                           occupancy};
      metrics.UpdateAll(current_metrics);
      Reporter::PrintResultsData(current_metrics, stats_res->NumSamples());

      if (num_blocks * block_size > kernel_info.n) {
        break;
      }
    }
    PrintResults(kernel_info.name + " current", metrics);
  }
  PrintResults(kernel_info.name + " final", metrics);
  return metrics;
}

template <typename KernelFunc>
Metrics RunUnstridedSearch(
    cudaDeviceProp hardware_info,
    IKernel<KernelFunc> &ex) {  // NOLINT(runtime/references)
  Metrics metrics;

  int num_blocks, block_size;
  OptimizeOccupancy(hardware_info, num_blocks, block_size, ex.GetKernel());
  std::cout << "expected optimal num_blocks: " << num_blocks << std::endl;
  std::cout << "expected optimal block_size: " << block_size << std::endl;

  auto kernel_info = ex.GetKernelInfo();

  auto block_size_gen = ex.GetBlockSizeGenerator();
  while (auto block_size_opt = block_size_gen->Next()) {
    int block_size = *block_size_opt;
    int num_blocks = (ex.GetKernelInfo().n + block_size - 1) / block_size;
    {
      Reporter::PrintResultsHeader(num_blocks, block_size);
      auto occupancy =
          Occupancy(hardware_info, num_blocks, block_size, ex.GetKernel());

      auto stats_res =
          RepeatUntil(kRequiredPrecision, ex, num_blocks, block_size);

      if (!stats_res) {
        std::cout << " [failed]" << std::endl;
        continue;
      }
      auto mean_res = stats_res->EstimatedMean();
      if (!mean_res || 0.0 == *mean_res) {
        std::cout << " [failed, mean==0.0!]" << std::endl;
        continue;
      }
      auto time_in_ms = *mean_res;
      auto time_in_seconds = time_in_ms / 1000.0;
      auto bandwidth = kernel_info.total_bytes / time_in_seconds;
      Data current_metrics{num_blocks, block_size, time_in_ms, bandwidth,
                           occupancy};
      metrics.UpdateAll(current_metrics);
      Reporter::PrintResultsData(current_metrics, stats_res->NumSamples());
    }
  }
  PrintResults(kernel_info.name + " final", metrics);
  return metrics;
}
