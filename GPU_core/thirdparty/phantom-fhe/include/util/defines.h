#pragma once

constexpr int MOD_BIT_COUNT_MAX = 61;
constexpr int MOD_BIT_COUNT_MIN = 2;

constexpr int INTERNAL_MOD_BIT_COUNT = 61;

constexpr int USER_MOD_BIT_COUNT_MAX = 60;
constexpr int USER_MOD_BIT_COUNT_MIN = 2;

constexpr int PLAIN_MOD_BIT_COUNT_MAX = USER_MOD_BIT_COUNT_MAX;
constexpr int PLAIN_MOD_BIT_COUNT_MIN = USER_MOD_BIT_COUNT_MIN;

constexpr int COEFF_MOD_COUNT_MAX = 64;
constexpr int COEFF_MOD_COUNT_MIN = 1;

constexpr int POLY_MOD_DEGREE_MAX = 131072;
constexpr int POLY_MOD_DEGREE_MIN = 2;

constexpr int CIPHERTEXT_SIZE_MAX = 16;
constexpr int CIPHERTEXT_SIZE_MIN = 2;

#define MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (MOD_BIT_COUNT_MAX << 1)))

#include <cuda_runtime_api.h>
#include <cuda.h>
