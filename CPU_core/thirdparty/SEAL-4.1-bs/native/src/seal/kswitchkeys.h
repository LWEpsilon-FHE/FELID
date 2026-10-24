// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/publickey.h"
#include "seal/valcheck.h"
#include "seal/version.h"
#include <iostream>
#include <vector>

namespace seal
{
     
    class KSwitchKeys
    {
        friend class KeyGenerator;
        friend class RelinKeys;
        friend class GaloisKeys;

    public:
         
        KSwitchKeys() = default;

        KSwitchKeys(const KSwitchKeys &copy) = default;

        KSwitchKeys(KSwitchKeys &&source) = default;

        KSwitchKeys &operator=(const KSwitchKeys &assign);

        KSwitchKeys &operator=(KSwitchKeys &&assign) = default;

        SEAL_NODISCARD inline std::size_t size() const noexcept
        {
            return std::accumulate(keys_.cbegin(), keys_.cend(), std::size_t(0), [](std::size_t res, auto &next_key) {
                return res + (next_key.empty() ? 0 : 1);
            });
        }

        SEAL_NODISCARD inline auto &data() noexcept
        {
            return keys_;
        }

        SEAL_NODISCARD inline auto &data() const noexcept
        {
            return keys_;
        }

        SEAL_NODISCARD inline auto &data(std::size_t index)
        {
            if (index >= keys_.size() || keys_[index].empty())
            {
                throw std::invalid_argument("keyswitching key does not exist");
            }
            return keys_[index];
        }

        SEAL_NODISCARD inline const auto &data(std::size_t index) const
        {
            if (index >= keys_.size() || keys_[index].empty())
            {
                throw std::invalid_argument("keyswitching key does not exist");
            }
            return keys_[index];
        }

        SEAL_NODISCARD inline auto &parms_id() noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline auto &parms_id() const noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            std::size_t total_key_size = util::mul_safe(keys_.size(), sizeof(std::uint64_t));  
            for (auto &key_dim1 : keys_)
            {
                for (auto &key_dim2 : key_dim1)
                {
                    total_key_size = util::add_safe(
                        total_key_size, util::safe_cast<std::size_t>(key_dim2.save_size(compr_mode_type::none)));
                }
            }

            std::size_t members_size = Serialization::ComprSizeEstimate(
                util::add_safe(
                    sizeof(parms_id_),
                    sizeof(std::uint64_t),  
                    total_key_size),
                compr_mode);

            return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&KSwitchKeys::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&KSwitchKeys::load_members, this, context, _1, _2), stream, false);
        }

        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            KSwitchKeys new_keys;
            new_keys.pool_ = pool_;
            auto in_size = new_keys.unsafe_load(context, stream);
            if (!is_valid_for(new_keys, context))
            {
                throw std::logic_error("KSwitchKeys data is invalid");
            }
            std::swap(*this, new_keys);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&KSwitchKeys::save_members, this, _1), save_size(compr_mode_type::none), out, size,
                compr_mode, false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&KSwitchKeys::load_members, this, context, _1, _2), in, size, false);
        }

        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            KSwitchKeys new_keys;
            new_keys.pool_ = pool_;
            auto in_size = new_keys.unsafe_load(context, in, size);
            if (!is_valid_for(new_keys, context))
            {
                throw std::logic_error("KSwitchKeys data is invalid");
            }
            std::swap(*this, new_keys);
            return in_size;
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return pool_;
        }

    private:
        void save_members(std::ostream &stream) const;

        void load_members(const SEALContext &context, std::istream &stream, SEALVersion version);

        MemoryPoolHandle pool_ = MemoryManager::GetPool();

        parms_id_type parms_id_ = parms_id_zero;

        std::vector<std::vector<PublicKey>> keys_{};
    };
}
