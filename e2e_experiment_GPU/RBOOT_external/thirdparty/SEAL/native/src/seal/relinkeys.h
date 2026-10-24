// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/kswitchkeys.h"
#include "seal/memorymanager.h"
#include "seal/util/defines.h"
#include <vector>

namespace seal
{
     
    class RelinKeys : public KSwitchKeys
    {
    public:
         
        SEAL_NODISCARD inline static std::size_t get_index(std::size_t key_power)
        {
            if (key_power < 2)
            {
                throw std::invalid_argument("key_power cannot be less than 2");
            }
            return key_power - 2;
        }

        SEAL_NODISCARD inline bool has_key(std::size_t key_power) const
        {
            std::size_t index = get_index(key_power);
            return data().size() > index && !data()[index].empty();
        }

        SEAL_NODISCARD inline auto &key(std::size_t key_power) const
        {
            return KSwitchKeys::data(get_index(key_power));
        }
    };
}
