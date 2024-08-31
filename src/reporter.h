#pragma once

#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

class Reporter {
 public:
  // Return the given value as a string formatted with SI prefixes. Values
  // will be formatted with two decimal places.
  static std::string FormatToSI(double number);

  // Return the given integer formatted with commas, like "1,024,000".
  static std::string FormatWithCommas(int n);

  // Print the timing results header.
  inline static void PrintResultsHeader(int num_blocks, int block_size) {
    std::cout << "<<numBlocks, blockSize>> = <<" << std::setw(10)
              << Reporter::FormatWithCommas(num_blocks) << ", " << std::setw(5)
              << Reporter::FormatWithCommas(block_size) << ">>";
  }

  // Print the timing results data.
  inline static void PrintResultsData(double bandwidth, double time,
                                      std::optional<int> num_samples) {
    std::cout << ", bandwidth: " << std::setw(10)
              << Reporter::FormatToSI(bandwidth) << "B/s";

    std::cout << ", time: " << std::fixed << std::setprecision(2)
              << std::setw(7) << time << " ms";

    if (num_samples.has_value()) {
      std::cout << " (over " << *num_samples << " runs) ";
    }

    std::cout << std::endl;
  }

  // Print the complete timing results.
  inline static void PrintResults(std::string prefix, int num_blocks,
                                  int block_size, double bandwidth,
                                  double time) {
    std::cout << prefix;
    Reporter::PrintResultsHeader(num_blocks, block_size);
    PrintResultsData(bandwidth, time, std::nullopt);
  }
};
