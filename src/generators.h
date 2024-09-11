#pragma once

#include <optional>

namespace cuda_optimizer {

// Generators are used to create increments for grid searches.
class IGridSizeGenerator {
 public:
  virtual std::optional<int> Next() = 0;
};

// Generates values, such as for num_blocks, by doubling the number of blocks up
// to a maximum.
class DoublingGenerator : public IGridSizeGenerator {
 public:
  explicit DoublingGenerator(int mnb) : max_num_blocks_(mnb) {}
  std::optional<int> Next() override {
    if (current_ > max_num_blocks_) {
      return std::nullopt;
    }
    int result = current_;
    current_ *= 2;
    return result;
  }

 private:
  int current_ = 1;
  const int max_num_blocks_;
};

// Generates values, such as for block_size, by incrementing by 32 up to a
// maximum.
class IncrementBy32Generator : public IGridSizeGenerator {
 public:
  explicit IncrementBy32Generator(int mbs) : max_block_size_(mbs) {}
  std::optional<int> Next() override {
    if (block_size_ > max_block_size_) {
      return std::nullopt;
    }
    int result = block_size_;
    i_++;
    block_size_ = 32 * i_;
    return result;
  }

 private:
  int i_ = 0;
  int block_size_ = 1;
  const int max_block_size_;
};

}  // namespace cuda_optimizer
