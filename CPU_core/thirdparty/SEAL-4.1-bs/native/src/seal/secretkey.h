// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/plaintext.h"
#include "seal/randomgen.h"
#include "seal/valcheck.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>

namespace seal
{
     
    class SecretKey
    {
        friend class KeyGenerator;

    public:
         
        SecretKey() = default;

        SecretKey(const SecretKey &copy)
        {

            sk_ = copy.sk_;
        }

        ~SecretKey() = default;

        SecretKey(SecretKey &&source) = default;

        SecretKey &operator=(const SecretKey &assign)
        {
            Plaintext new_sk(MemoryManager::GetPool(mm_prof_opt::mm_force_new, true));
            new_sk = assign.sk_;
            std::swap(sk_, new_sk);
            return *this;
        }

        SecretKey &operator=(SecretKey &&assign) = default;

        SEAL_NODISCARD inline auto &data() noexcept
        {
            return sk_;
        }

        SEAL_NODISCARD inline auto &data() const noexcept
        {
            return sk_;
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return sk_.save_size(compr_mode);
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Plaintext::save_members, &sk_, _1), sk_.save_size(compr_mode_type::none), stream, compr_mode,
                true);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            using namespace std::placeholders;

            Plaintext new_sk(MemoryManager::GetPool(mm_prof_opt::mm_force_new, true));
            auto in_size = Serialization::Load(
                std::bind(&Plaintext::load_members, &new_sk, std::move(context), _1, _2), stream, true);
            std::swap(sk_, new_sk);
            return in_size;
        }

        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            SecretKey new_sk;
            auto in_size = new_sk.unsafe_load(context, stream);
            if (!is_valid_for(new_sk, context))
            {
                throw std::logic_error("SecretKey data is invalid");
            }
            std::swap(*this, new_sk);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Plaintext::save_members, &sk_, _1), sk_.save_size(compr_mode_type::none), out, size,
                compr_mode, true);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;

            Plaintext new_sk(MemoryManager::GetPool(mm_prof_opt::mm_force_new, true));
            auto in_size = Serialization::Load(
                std::bind(&Plaintext::load_members, &new_sk, std::move(context), _1, _2), in, size, true);
            std::swap(sk_, new_sk);
            return in_size;
        }

        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            SecretKey new_sk;
            auto in_size = new_sk.unsafe_load(context, in, size);
            if (!is_valid_for(new_sk, context))
            {
                throw std::logic_error("SecretKey data is invalid");
            }
            std::swap(*this, new_sk);
            return in_size;
        }

        SEAL_NODISCARD inline auto &parms_id() noexcept
        {
            return sk_.parms_id();
        }

        SEAL_NODISCARD inline auto &parms_id() const noexcept
        {
            return sk_.parms_id();
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return sk_.pool();
        }

    private:
         
        Plaintext sk_{ MemoryManager::GetPool(mm_prof_opt::mm_force_new, true) };
    };
}
