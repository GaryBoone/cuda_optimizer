
#include "./reporter.h"

#include <iostream>
#include <locale>
#include <string>

std::string Reporter::FormatToSI(double number) {
  struct Scale {
    double divisor;
    char prefix;
  };

  const Scale scales[] = {
      {1e24, 'Y'}, {1e21, 'Z'}, {1e18, 'E'}, {1e15, 'P'}, {1e12, 'T'},
      {1e9, 'G'},  {1e6, 'M'},  {1e3, 'K'},  {1, ' '}  // No prefix if < 1000.
  };

  Scale selected_scale = {1, ' '};
  for (const auto &scale : scales) {
    if (number >= scale.divisor) {
      selected_scale = scale;
      break;
    }
  }

  double scaled_number = number / selected_scale.divisor;
  char formatted_string[50];

  if (selected_scale.prefix == ' ') {
    snprintf(formatted_string, sizeof(formatted_string), "%.2f", scaled_number);
  } else {
    snprintf(formatted_string, sizeof(formatted_string), "%.2f%c",
             scaled_number, selected_scale.prefix);
  }

  return std::string(formatted_string);
}

std::string Reporter::FormatWithCommas(int n) {
  std::string result = std::to_string(n);
  for (int i = result.size() - 3; i > 0; i -= 3) {
    result.insert(i, ",");
  }
  return result;
}
