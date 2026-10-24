#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace moai {
using namespace std;
using namespace std::chrono;

inline void is_same_stream(const cudaStream_t s1,
                           const cudaStream_t s2) {
    if (s1 != s2) {
        throw std::invalid_argument("data are not in the same stream!");
    }
}
}
