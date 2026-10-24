#pragma once

#include <cstring>
#include <gputype.h>
#include <random>

void inline random_bytes(unsigned char *buf, size_t count, const cudaStream_t &stream) {
    std::random_device rd;
    std::uniform_int_distribution<unsigned short> dist(std::numeric_limits<unsigned char>::min(),
                                                       std::numeric_limits<unsigned char>::max());
    std::vector<uint8_t> temp(count);
    for (auto &i: temp) {
        i = dist(rd);
    }
    cudaMemcpyAsync(buf, temp.data(), count, cudaMemcpyHostToDevice, stream);
}

__host__ __device__ inline uint32_t load_littleendian(const unsigned char *x) {
    return (uint32_t) (x[0]) | (((uint32_t) (x[1])) << 8) | (((uint32_t) (x[2])) << 16) | (((uint32_t) (x[3])) << 24);
}

__host__ __device__ inline uint32_t rotate(const uint32_t u, const int c) {
    return (u << c) | (u >> (32 - c));
}

__global__ void sample_ternary_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size);

__global__ void sample_ternary_poly_with_zero(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size, const int zero_threshold);

__global__ void sample_uniform_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                    const uint64_t poly_degree, const uint64_t coeff_mod_size);

__global__ void sample_error_poly(uint64_t *out, const uint8_t *prng_seed, const DModulus *modulus,
                                  const uint64_t poly_degree, const uint64_t coeff_mod_size);
