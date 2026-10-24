// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/kswitchkeys.h"
#include "seal/memorymanager.h"
#include "seal/util/defines.h"
#include "seal/util/galois.h"
#include <vector>

namespace seal
{
     
    class GaloisKeys : public KSwitchKeys
    {
    public:
         
        SEAL_NODISCARD inline static std::size_t get_index(std::uint32_t galois_elt)
        {
            return util::GaloisTool::GetIndexFromElt(galois_elt);
        }

        SEAL_NODISCARD inline bool has_key(std::uint32_t galois_elt) const
        {
            std::size_t index = get_index(galois_elt);
            return data().size() > index && !data()[index].empty();
        }

        SEAL_NODISCARD inline const auto &key(std::uint32_t galois_elt) const
        {
            return KSwitchKeys::data(get_index(galois_elt));
        }
    };
}
