#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

inline void validate_layernorm_shape(std::size_t left, std::size_t right,
    std::size_t slots, int row, int col, int batch_size,
    const std::vector<double>& gamma, const std::vector<double>& beta, double c) {
  if (row <= 0 || col <= 0 || batch_size <= 0 || !std::isfinite(c) || c <= 0 ||
      left == 0 || left != right || gamma.size() != static_cast<std::size_t>(col) ||
      beta.size() != static_cast<std::size_t>(col))
    throw std::invalid_argument("Invalid LayerNorm dimensions or coefficients");
  const std::size_t block = static_cast<std::size_t>(row) * batch_size;
  if (block > slots || slots % block ||
      (block & (block - 1)) || left * (slots / block) != static_cast<std::size_t>(col))
    throw std::invalid_argument("Invalid LayerNorm packing");
}

inline std::size_t layernorm_feature(std::size_t ciphertext, std::size_t slot,
                                    std::size_t slots, int row, int batch_size) {
  const std::size_t block = static_cast<std::size_t>(row) * batch_size;
  return ciphertext * (slots / block) + slot / block;
}
