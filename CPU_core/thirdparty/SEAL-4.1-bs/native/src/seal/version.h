// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/util/defines.h"
#include <cstdint>

namespace seal
{
     
    struct SEALVersion
    {
         
        std::uint8_t major = SEAL_VERSION_MAJOR;

        std::uint8_t minor = SEAL_VERSION_MINOR;

        std::uint8_t patch = SEAL_VERSION_PATCH;

        std::uint8_t tweak = 0;
    };
}
