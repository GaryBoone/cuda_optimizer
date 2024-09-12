#pragma once

#include <optional>
#include <string>

#include "./metrics.h"

namespace cuda_optimizer {

class Reporter {
 public:
  // Return the given value as a string formatted with SI prefixes. Values
  // will be formatted with two decimal places.
  static std::string FormatToSI(double number);

  // Return the given integer formatted with commas, like "1,024,000".
  static std::string FormatWithCommas(int n);

  // Print the timing results header.
  static void PrintResultsHeader(int num_blocks, int block_size);

  // Print the timing results data.
  static void PrintResultsData(Data data, std::optional<int> num_samples);

  // Print the complete timing results.
  static void PrintResults(std::string prefix, Data data);
};

}  // namespace cuda_optimizer
