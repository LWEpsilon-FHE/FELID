// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "hestdparms.h"
#include "common.h"
#include "modulus.h"
#include "cuda_wrapper.cuh"

namespace phantom::util::global_variables {
     
    constexpr std::size_t prng_seed_uint64_count = 8;
    constexpr std::size_t prng_seed_byte_count = prng_seed_uint64_count * arith::bytes_per_uint64;

    constexpr double noise_standard_deviation = distributionParameter;

    constexpr double noise_distribution_width_multiplier = 6;

    constexpr double noise_max_deviation = noise_standard_deviation * noise_distribution_width_multiplier;

    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus128();

    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus192();

    const std::map<std::size_t, std::vector<arith::Modulus>> &GetDefaultCoeffModulus256();

    extern std::unique_ptr<phantom::util::cuda_stream_wrapper> default_stream;
}
