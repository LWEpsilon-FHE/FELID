// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.h"
#include "seal/util/defines.h"

namespace seal
{
    class Plaintext;
    class Ciphertext;
    class SecretKey;
    class PublicKey;
    class KSwitchKeys;
    class RelinKeys;
    class GaloisKeys;

    SEAL_NODISCARD bool is_metadata_valid_for(
        const Plaintext &in, const SEALContext &context, bool allow_pure_key_levels = false);

    SEAL_NODISCARD bool is_metadata_valid_for(
        const Ciphertext &in, const SEALContext &context, bool allow_pure_key_levels = false);

    SEAL_NODISCARD bool is_metadata_valid_for(const SecretKey &in, const SEALContext &context);

    SEAL_NODISCARD bool is_metadata_valid_for(const PublicKey &in, const SEALContext &context);

    SEAL_NODISCARD bool is_metadata_valid_for(const KSwitchKeys &in, const SEALContext &context);

    SEAL_NODISCARD bool is_metadata_valid_for(const RelinKeys &in, const SEALContext &context);

    SEAL_NODISCARD bool is_metadata_valid_for(const GaloisKeys &in, const SEALContext &context);

    SEAL_NODISCARD bool is_buffer_valid(const Plaintext &in);

    SEAL_NODISCARD bool is_buffer_valid(const Ciphertext &in);

    SEAL_NODISCARD bool is_buffer_valid(const SecretKey &in);

    SEAL_NODISCARD bool is_buffer_valid(const PublicKey &in);

    SEAL_NODISCARD bool is_buffer_valid(const KSwitchKeys &in);

    SEAL_NODISCARD bool is_buffer_valid(const RelinKeys &in);

    SEAL_NODISCARD bool is_buffer_valid(const GaloisKeys &in);

    SEAL_NODISCARD bool is_data_valid_for(const Plaintext &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const Ciphertext &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const SecretKey &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const PublicKey &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const KSwitchKeys &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const RelinKeys &in, const SEALContext &context);

    SEAL_NODISCARD bool is_data_valid_for(const GaloisKeys &in, const SEALContext &context);

    SEAL_NODISCARD inline bool is_valid_for(const Plaintext &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const Ciphertext &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const SecretKey &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const PublicKey &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const KSwitchKeys &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const RelinKeys &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }

    SEAL_NODISCARD inline bool is_valid_for(const GaloisKeys &in, const SEALContext &context)
    {
        return is_buffer_valid(in) && is_data_valid_for(in, context);
    }
}
