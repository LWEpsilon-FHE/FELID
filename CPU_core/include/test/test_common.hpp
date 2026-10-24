#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

class ErrorStats {
  std::size_t count_ = 0, failures_ = 0;
  double mean_ = 0, m2_ = 0, maximum_ = 0;
  double atol_, rtol_;
public:
  explicit ErrorStats(double atol = 1e-3, double rtol = 1e-5)
      : atol_(atol), rtol_(rtol) {}
  void add(double actual, double expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected))
      throw std::runtime_error("Non-finite test result");
    const double error = actual - expected, magnitude = std::abs(error);
    if (!std::isfinite(error)) throw std::runtime_error("Non-finite test error");
    const double delta = error - mean_;
    mean_ += delta / ++count_;
    m2_ += delta * (error - mean_);
    maximum_ = std::max(maximum_, magnitude);
    if (magnitude > atol_ + rtol_ * std::abs(expected)) {
      if (failures_ < 3)
        std::cerr << "Failed sample " << count_ - 1 << ": actual=" << std::setprecision(12)
                  << actual << ", reference=" << expected << ", absolute error=" << magnitude << '\n';
      ++failures_;
    }
  }
  void report() const {
    if (!count_) throw std::runtime_error("No test samples");
    const auto precision = std::cout.precision();
    std::cout << std::setprecision(12)
              << "samples = " << count_ << '\n'
              << "average error = " << mean_ << '\n'
              << "max absolute error = " << maximum_ << '\n'
              << "variance of errors = " << m2_ / count_ << '\n'
              << "tolerance = " << atol_ << " + " << rtol_ << " * abs(reference)\n"
              << "failed samples = " << failures_ << std::endl;
    std::cout.precision(precision);
    if (failures_) throw std::runtime_error("Numerical tolerance exceeded");
  }
};
