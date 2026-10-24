// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/hash.h"

using namespace std;

namespace seal
{
    namespace util
    {

        constexpr size_t HashFunction::hash_block_uint64_count;

        constexpr size_t HashFunction::hash_block_byte_count;

        constexpr HashFunction::hash_block_type HashFunction::hash_zero_block;
    }  
}
