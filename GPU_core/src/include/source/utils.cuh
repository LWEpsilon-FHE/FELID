#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace moai {
using namespace std;
using namespace std::chrono;

class Timer {
 private:
  std::chrono::_V2::system_clock::time_point start_;
  std::chrono::_V2::system_clock::time_point end_;

  template <typename T = milliseconds>
  inline void print_duration(const char *text) {
    cout << text << duration_cast<T>(end_ - start_).count() << endl;
  }

 public:
  Timer() {
    start_ = high_resolution_clock::now();
  }

  inline void start() {
    start_ = high_resolution_clock::now();
  }

  template <typename T = milliseconds>
  inline void stop(const char *text) {
    end_ = high_resolution_clock::now();
    print_duration<T>(text);
  }

  inline void stop() {
    end_ = high_resolution_clock::now();
  }

  template <typename T = milliseconds>
  inline long duration() {
    return duration_cast<T>(end_ - start_).count() / 1.0;
  }
};

inline void append_csv_row(const std::string& path,
                           const std::string& name,
                           double time) {
  bool need_header = !std::filesystem::exists(path);
  std::ofstream ofs(path, std::ios::app);
  if (!ofs) {
    std::cerr << "Failed to open CSV: " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (need_header) {
    ofs << "name,time(s)\n";
  }
  ofs << std::fixed << std::setprecision(3)
      << name << ","
      << time << "\n";
}


inline void is_same_stream(const cudaStream_t s1,
                           const cudaStream_t s2) {
    if (s1 != s2) {
        throw std::invalid_argument("data are not in the same stream!");
    }
}
}  // namespace moai
