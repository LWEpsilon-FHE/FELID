// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/util/hestdparms.h"
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace seal
{
    class Modulus;

    namespace util
    {
        class MemoryPool;

        namespace global_variables
        {
            extern std::shared_ptr<MemoryPool> const global_memory_pool;

#ifndef _M_CEE
            extern thread_local std::shared_ptr<MemoryPool> const tls_memory_pool;
#endif
             
            constexpr double noise_standard_deviation = seal_he_std_parms_error_std_dev;

            constexpr double noise_distribution_width_multiplier = 6;

            constexpr double noise_max_deviation = noise_standard_deviation * noise_distribution_width_multiplier;

            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus128();

            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus192();

            const std::map<std::size_t, std::vector<Modulus>> &GetDefaultCoeffModulus256();
        }  
    }  
}
