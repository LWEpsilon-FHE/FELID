// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/randomgen.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace seal
{
     
    class RandomToStandardAdapter
    {
    public:
        using result_type = std::uint32_t;

        RandomToStandardAdapter(std::shared_ptr<UniformRandomGenerator> generator) : generator_(generator)
        {
            if (!generator_)
            {
                throw std::invalid_argument("generator cannot be null");
            }
        }

        SEAL_NODISCARD inline result_type operator()()
        {
            return generator_->generate();
        }

        SEAL_NODISCARD inline auto generator() const noexcept
        {
            return generator_;
        }

        SEAL_NODISCARD inline static constexpr result_type min() noexcept
        {
            return std::numeric_limits<result_type>::min();
        }

        SEAL_NODISCARD static constexpr result_type max() noexcept
        {
            return std::numeric_limits<result_type>::max();
        }

    private:
        std::shared_ptr<UniformRandomGenerator> generator_;
    };
}
