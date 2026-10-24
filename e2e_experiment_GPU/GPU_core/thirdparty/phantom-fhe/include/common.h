#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
 
#include <utility>
#include <vector>
#include <stdio.h>

#define SWITCH_POINT 2048
#define MAX_THREAD_PER_BLOCK 1024
#define NTT_THREAD_PER_BLOCK 1024

namespace phantom {
    namespace util {
         
        constexpr dim3 blockDimGlb(128);

        constexpr dim3 gridDimNTT(4096);
        constexpr dim3 blockDimNTT(128);
         
        constexpr size_t per_thread_sample_size = 8;

        constexpr size_t per_block_pad = 4;

        constexpr double two_pow_64 = 18446744073709551616.0;

        constexpr int n_cuda_streams = 10;

        __device__ __constant__ constexpr double two_pow_64_dev = 18446744073709551616.0;

        __device__ __constant__ constexpr int bytes_per_uint64_dev = sizeof(std::uint64_t);

        __device__ __constant__ constexpr int bits_per_nibble_dev = 4;

        __device__ __constant__ constexpr int bits_per_byte_dev = 8;

        __device__ __constant__ constexpr int bits_per_uint64_dev = bytes_per_uint64_dev * bits_per_byte_dev;
    }
}
