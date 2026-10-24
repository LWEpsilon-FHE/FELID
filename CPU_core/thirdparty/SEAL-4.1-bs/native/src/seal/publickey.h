// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/valcheck.h"
#include <iostream>

namespace seal
{
     
    class PublicKey
    {
        friend class KeyGenerator;
        friend class KSwitchKeys;

    public:
         
        PublicKey() = default;

        PublicKey(const PublicKey &copy) = default;

        PublicKey(PublicKey &&source) = default;

        PublicKey &operator=(const PublicKey &assign) = default;

        PublicKey &operator=(PublicKey &&assign) = default;

        SEAL_NODISCARD inline auto &data() noexcept
        {
            return pk_;
        }

        SEAL_NODISCARD inline auto &data() const noexcept
        {
            return pk_;
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return pk_.save_size(compr_mode);
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return pk_.save(stream, compr_mode);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            Ciphertext new_pk(pk_.pool());
            auto in_size = new_pk.unsafe_load(context, stream);
            std::swap(pk_, new_pk);
            return in_size;
        }

        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            PublicKey new_pk(pool());
            auto in_size = new_pk.unsafe_load(context, stream);
            if (!is_valid_for(new_pk, context))
            {
                throw std::logic_error("PublicKey data is invalid");
            }
            std::swap(*this, new_pk);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return pk_.save(out, size, compr_mode);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            Ciphertext new_pk(pk_.pool());
            auto in_size = new_pk.unsafe_load(context, in, size);
            std::swap(pk_, new_pk);
            return in_size;
        }

        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            PublicKey new_pk(pool());
            auto in_size = new_pk.unsafe_load(context, in, size);
            if (!is_valid_for(new_pk, context))
            {
                throw std::logic_error("PublicKey data is invalid");
            }
            std::swap(*this, new_pk);
            return in_size;
        }

        SEAL_NODISCARD inline auto &parms_id() noexcept
        {
            return pk_.parms_id();
        }

        SEAL_NODISCARD inline auto &parms_id() const noexcept
        {
            return pk_.parms_id();
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return pk_.pool();
        }

        struct PublicKeyPrivateHelper;

    private:
         
        PublicKey(MemoryPoolHandle pool) : pk_(std::move(pool))
        {}

        Ciphertext pk_;
    };
}
