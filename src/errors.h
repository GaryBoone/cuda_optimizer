#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tl/expected.hpp"

namespace cuda_optimizer {

struct ErrorInfo {
  enum ErrorType {
    kNone,
    kInvalidDegreesOfFreedom,
    kTooFewSamples,
    kDivisionByZero,
    kUnexpectedKernelResult
  } error_type;

  std::string message;

  explicit ErrorInfo(ErrorType t) : error_type(t), message("unknown error") {}

  ErrorInfo(ErrorType t, std::string m)
      : error_type(t), message(std::move(m)) {}
};

using ExpectedDouble = tl::expected<double, ErrorInfo>;
using ExpectedInt = tl::expected<int, ErrorInfo>;
using ExpectedBool = tl::expected<bool, ErrorInfo>;
using ExpectedVoid = tl::expected<void, ErrorInfo>;

}  // namespace cuda_optimizer
