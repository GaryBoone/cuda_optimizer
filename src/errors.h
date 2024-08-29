#pragma once

#include <optional>
#include <string>
#include <vector>

#include "tl/expected.hpp"

struct ErrorInfo {
  enum ErrorType {
    kNone,
    kInvalidDegreesOfFreedom,
    kTooFewSamples,
    kDivisionByZero
  } error_type;

  std::string message;

  ErrorInfo(ErrorType t) : error_type(t), message("unknown error") {}

  ErrorInfo(ErrorType t, std::string m)
      : error_type(t), message(std::move(m)) {}
};

using ExpectedDouble = tl::expected<double, ErrorInfo>;
using ExpectedBool = tl::expected<bool, ErrorInfo>;
using ExpectedVoid = tl::expected<void, ErrorInfo>;
