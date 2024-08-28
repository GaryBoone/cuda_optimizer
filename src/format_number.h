#pragma once

#include <string>

// Return the given value as a string formatted with SI prefixes. Values
// will be formatted with two decimal places.
inline std::string formatNumber(double number) {
  struct Scale {
    double divisor;
    char prefix;
  };

  // Define the scales and their corresponding prefixes
  const Scale scales[] = {
      {1e24, 'Y'}, {1e21, 'Z'}, {1e18, 'E'}, {1e15, 'P'}, {1e12, 'T'},
      {1e9, 'G'},  {1e6, 'M'},  {1e3, 'K'},  {1, ' '} // No prefix if <  1000.
  };

  // Determine the appropriate scale
  Scale selectedScale = {1, ' '};
  for (const auto &scale : scales) {
    if (number >= scale.divisor) {
      selectedScale = scale;
      break;
    }
  }

  // Scale the number and format it
  double scaledNumber = number / selectedScale.divisor;
  char formattedString[50]; // Buffer to hold the formatted string

  // Use snprintf for formatting to control the precision and size
  if (selectedScale.prefix == ' ') {
    snprintf(formattedString, sizeof(formattedString), "%.2f", scaledNumber);
  } else {
    snprintf(formattedString, sizeof(formattedString), "%.2f%c", scaledNumber,
             selectedScale.prefix);
  }

  return std::string(formattedString);
}
