// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/randomgen.h"
#include "seal/serialization.h"
#include "seal/version.h"
#include "seal/util/defines.h"
#include "seal/util/globals.h"
#include "seal/util/hash.h"
#include "seal/util/ztools.h"
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

namespace seal
{
     
    enum class scheme_type : std::uint8_t
    {
         
        none = 0x0,

        bfv = 0x1,

        ckks = 0x2,

        bgv = 0x3
    };

    using parms_id_type = util::HashFunction::hash_block_type;

    extern const parms_id_type parms_id_zero;

    class EncryptionParameters
    {
        friend struct std::hash<EncryptionParameters>;

    public:
         
        EncryptionParameters(scheme_type scheme = scheme_type::none) : scheme_(scheme)
        {
            compute_parms_id();
        }

        EncryptionParameters(std::uint8_t scheme)
        {
             
            if (!is_valid_scheme(scheme))
            {
                throw std::invalid_argument("unsupported scheme");
            }

            scheme_ = static_cast<scheme_type>(scheme);
            compute_parms_id();
        }

        EncryptionParameters(const EncryptionParameters &copy) = default;

        EncryptionParameters &operator=(const EncryptionParameters &assign) = default;

        EncryptionParameters(EncryptionParameters &&source) = default;

        EncryptionParameters &operator=(EncryptionParameters &&assign) = default;

        inline void set_poly_modulus_degree(std::size_t poly_modulus_degree)
        {
            if (scheme_ == scheme_type::none && poly_modulus_degree)
            {
                throw std::logic_error("poly_modulus_degree is not supported for this scheme");
            }

            poly_modulus_degree_ = poly_modulus_degree;

            compute_parms_id();
        }

        inline void set_secret_key_hamming_weight(std::size_t secret_key_hamming_weight)
        {
            if (scheme_ == scheme_type::none && secret_key_hamming_weight)
            {
                throw std::logic_error("secret key hamming weight is not supported for this scheme");
            }

            secret_key_hamming_weight_ = secret_key_hamming_weight;

            compute_parms_id();
        }

        inline void set_sparse_slots(std::size_t sparse_slots)
        {
            if (scheme_ == scheme_type::none && sparse_slots)
            {
                throw std::logic_error("secret key hamming weight is not supported for this scheme");
            }

            if ((sparse_slots & (sparse_slots - 1)) != 0)
            {
                throw std::logic_error("secret key hamming weight is not zero or power-of-two");
            }

            sparse_slots_= sparse_slots;

            compute_parms_id();
        }

        inline void set_coeff_modulus(const std::vector<Modulus> &coeff_modulus)
        {
             
            if (scheme_ == scheme_type::none)
            {
                if (!coeff_modulus.empty())
                {
                    throw std::logic_error("coeff_modulus is not supported for this scheme");
                }
            }
            else if (coeff_modulus.size() > SEAL_COEFF_MOD_COUNT_MAX || coeff_modulus.size() < SEAL_COEFF_MOD_COUNT_MIN)
            {
                throw std::invalid_argument("coeff_modulus is invalid");
            }

            coeff_modulus_ = coeff_modulus;

            compute_parms_id();
        }

        inline void set_plain_modulus(const Modulus &plain_modulus)
        {
             
            if (scheme_ != scheme_type::bfv && scheme_ != scheme_type::bgv && !plain_modulus.is_zero())
            {
                throw std::logic_error("plain_modulus is not supported for this scheme");
            }

            plain_modulus_ = plain_modulus;

            compute_parms_id();
        }

        inline void set_plain_modulus(std::uint64_t plain_modulus)
        {
            set_plain_modulus(Modulus(plain_modulus));
        }

        inline void set_random_generator(std::shared_ptr<UniformRandomGeneratorFactory> random_generator) noexcept
        {
            random_generator_ = std::move(random_generator);
        }

        SEAL_NODISCARD inline scheme_type scheme() const noexcept
        {
            return scheme_;
        }

        SEAL_NODISCARD inline std::size_t poly_modulus_degree() const noexcept
        {
            return poly_modulus_degree_;
        }

        SEAL_NODISCARD inline std::size_t secret_key_hamming_weight() const noexcept
        {
            return secret_key_hamming_weight_;
        }

        SEAL_NODISCARD inline std::size_t sparse_slots() const noexcept
        {
            return sparse_slots_;
        }

        SEAL_NODISCARD inline const std::vector<Modulus> &coeff_modulus() const noexcept
        {
            return coeff_modulus_;
        }

        SEAL_NODISCARD inline const Modulus &plain_modulus() const noexcept
        {
            return plain_modulus_;
        }

        SEAL_NODISCARD inline std::shared_ptr<UniformRandomGeneratorFactory> random_generator() const noexcept
        {
            return random_generator_;
        }

        SEAL_NODISCARD inline const parms_id_type &parms_id() const noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline bool operator==(const EncryptionParameters &other) const noexcept
        {
            return (parms_id_ == other.parms_id_);
        }

        SEAL_NODISCARD inline bool operator!=(const EncryptionParameters &other) const noexcept
        {
            return (parms_id_ != other.parms_id_);
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            std::size_t coeff_modulus_total_size =
                coeff_modulus_.empty()
                    ? std::size_t(0)
                    : util::safe_cast<std::size_t>(coeff_modulus_[0].save_size(compr_mode_type::none));
            coeff_modulus_total_size = util::mul_safe(coeff_modulus_total_size, coeff_modulus_.size());

            std::size_t members_size = Serialization::ComprSizeEstimate(
                util::add_safe(
                    sizeof(scheme_),
                    sizeof(std::uint64_t),  
                    sizeof(std::uint64_t),  
                    coeff_modulus_total_size,
                    util::safe_cast<std::size_t>(plain_modulus_.save_size(compr_mode_type::none))),
                compr_mode);

            return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&EncryptionParameters::save_members, this, _1), save_size(compr_mode_type::none), stream,
                compr_mode, false);
        }

        inline std::streamoff load(std::istream &stream)
        {
            using namespace std::placeholders;
            EncryptionParameters new_parms(scheme_type::none);
            auto in_size =
                Serialization::Load(std::bind(&EncryptionParameters::load_members, &new_parms, _1, _2), stream, false);
            std::swap(*this, new_parms);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&EncryptionParameters::save_members, this, _1), save_size(compr_mode_type::none), out, size,
                compr_mode, false);
        }

        inline std::streamoff load(const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            EncryptionParameters new_parms(scheme_type::none);
            auto in_size = Serialization::Load(
                std::bind(&EncryptionParameters::load_members, &new_parms, _1, _2), in, size, false);
            std::swap(*this, new_parms);
            return in_size;
        }

        struct EncryptionParametersPrivateHelper;

    private:
         
        SEAL_NODISCARD bool is_valid_scheme(std::uint8_t scheme) const noexcept
        {
            switch (scheme)
            {
            case static_cast<std::uint8_t>(scheme_type::none):

            case static_cast<std::uint8_t>(scheme_type::bfv):

            case static_cast<std::uint8_t>(scheme_type::ckks):

            case static_cast<std::uint8_t>(scheme_type::bgv):
                return true;
            }
            return false;
        }

        void compute_parms_id();

        void save_members(std::ostream &stream) const;

        void load_members(std::istream &stream, SEALVersion version);

        MemoryPoolHandle pool_ = MemoryManager::GetPool();

        scheme_type scheme_;

        std::size_t poly_modulus_degree_ = 0;

        std::size_t secret_key_hamming_weight_ = 0;

        std::size_t sparse_slots_ = 0;

        std::vector<Modulus> coeff_modulus_{};

        std::shared_ptr<UniformRandomGeneratorFactory> random_generator_{ nullptr };

        Modulus plain_modulus_{};

        parms_id_type parms_id_ = parms_id_zero;
    };
}  

namespace std
{
    template <>
    struct hash<seal::parms_id_type>
    {
        std::size_t operator()(const seal::parms_id_type &parms_id) const
        {
            std::uint64_t result = 17;
            result = 31 * result + parms_id[0];
            result = 31 * result + parms_id[1];
            result = 31 * result + parms_id[2];
            result = 31 * result + parms_id[3];
            return static_cast<std::size_t>(result);
        }
    };

    template <>
    struct hash<seal::EncryptionParameters>
    {
        std::size_t operator()(const seal::EncryptionParameters &parms) const
        {
            hash<seal::parms_id_type> parms_id_hash;
            return parms_id_hash(parms.parms_id_);
        }
    };
}
