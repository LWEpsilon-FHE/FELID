// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/serialization.h"
#include "seal/version.h"
#include "seal/util/defines.h"
#include "seal/util/hestdparms.h"
#include "seal/util/uintcore.h"
#include "seal/util/ztools.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace seal
{
     
    class Modulus
    {
    public:
         
        Modulus(std::uint64_t value = 0)
        {
            set_value(value);
        }

        Modulus(const Modulus &copy) = default;

        Modulus(Modulus &&source) = default;

        Modulus &operator=(const Modulus &assign) = default;

        Modulus &operator=(Modulus &&assign) = default;

        inline Modulus &operator=(std::uint64_t value)
        {
            set_value(value);
            return *this;
        }

        SEAL_NODISCARD inline int bit_count() const noexcept
        {
            return bit_count_;
        }

        SEAL_NODISCARD inline std::size_t uint64_count() const noexcept
        {
            return uint64_count_;
        }

        SEAL_NODISCARD inline const uint64_t *data() const noexcept
        {
            return &value_;
        }

        SEAL_NODISCARD inline std::uint64_t value() const noexcept
        {
            return value_;
        }

        SEAL_NODISCARD inline auto &const_ratio() const noexcept
        {
            return const_ratio_;
        }

        SEAL_NODISCARD inline bool is_zero() const noexcept
        {
            return value_ == 0;
        }

        SEAL_NODISCARD inline bool is_prime() const noexcept
        {
            return is_prime_;
        }

        SEAL_NODISCARD inline bool operator==(const Modulus &compare) const noexcept
        {
            return value_ == compare.value_;
        }

        SEAL_NODISCARD inline bool operator==(std::uint64_t compare) const noexcept
        {
            return value_ == compare;
        }

        SEAL_NODISCARD inline bool operator!=(const Modulus &compare) const noexcept
        {
            return !operator==(compare);
        }

        SEAL_NODISCARD inline bool operator!=(std::uint64_t compare) const noexcept
        {
            return !operator==(compare);
        }

        SEAL_NODISCARD inline bool operator<(const Modulus &compare) const noexcept
        {
            return value_ < compare.value_;
        }

        SEAL_NODISCARD inline bool operator<(std::uint64_t compare) const noexcept
        {
            return value_ < compare;
        }

        SEAL_NODISCARD inline bool operator<=(const Modulus &compare) const noexcept
        {
            return value_ <= compare.value_;
        }

        SEAL_NODISCARD inline bool operator<=(std::uint64_t compare) const noexcept
        {
            return value_ <= compare;
        }

        SEAL_NODISCARD inline bool operator>(const Modulus &compare) const noexcept
        {
            return value_ > compare.value_;
        }

        SEAL_NODISCARD inline bool operator>(std::uint64_t compare) const noexcept
        {
            return value_ > compare;
        }

        SEAL_NODISCARD inline bool operator>=(const Modulus &compare) const noexcept
        {
            return value_ >= compare.value_;
        }

        SEAL_NODISCARD inline bool operator>=(std::uint64_t compare) const noexcept
        {
            return value_ >= compare;
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            std::size_t members_size = Serialization::ComprSizeEstimate(util::add_safe(sizeof(value_)), compr_mode);

            return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Modulus::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        inline std::streamoff load(std::istream &stream)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Modulus::load_members, this, _1, _2), stream, false);
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Modulus::save_members, this, _1), save_size(compr_mode_type::none), out, size, compr_mode,
                false);
        }

        inline std::streamoff load(const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Modulus::load_members, this, _1, _2), in, size, false);
        }

        SEAL_NODISCARD std::uint64_t reduce(std::uint64_t value) const;

    private:
        void set_value(std::uint64_t value);

        void save_members(std::ostream &stream) const;

        void load_members(std::istream &stream, SEALVersion version);

        std::uint64_t value_ = 0;

        std::array<std::uint64_t, 3> const_ratio_{ { 0, 0, 0 } };

        std::size_t uint64_count_ = 0;

        int bit_count_ = 0;

        bool is_prime_ = false;
    };

    enum class sec_level_type : int
    {
         
        none = 0,

        tc128 = 128,

        tc192 = 192,

        tc256 = 256
    };

    class CoeffModulus
    {
    public:
        CoeffModulus() = delete;

        SEAL_NODISCARD static constexpr int MaxBitCount(
            std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128) noexcept
        {
            switch (sec_level)
            {
            case sec_level_type::tc128:
                return util::seal_he_std_parms_128_tc(poly_modulus_degree);

            case sec_level_type::tc192:
                return util::seal_he_std_parms_192_tc(poly_modulus_degree);

            case sec_level_type::tc256:
                return util::seal_he_std_parms_256_tc(poly_modulus_degree);

            case sec_level_type::none:
                return (std::numeric_limits<int>::max)();

            default:
                return 0;
            }
        }

        SEAL_NODISCARD static std::vector<Modulus> BFVDefault(
            std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128);

        SEAL_NODISCARD static std::vector<Modulus> Create(std::size_t poly_modulus_degree, std::vector<int> bit_sizes);
    };

    class PlainModulus
    {
    public:
        PlainModulus() = delete;

        SEAL_NODISCARD static inline Modulus Batching(std::size_t poly_modulus_degree, int bit_size)
        {
            return CoeffModulus::Create(poly_modulus_degree, { bit_size })[0];
        }

        SEAL_NODISCARD static inline std::vector<Modulus> Batching(
            std::size_t poly_modulus_degree, std::vector<int> bit_sizes)
        {
            return CoeffModulus::Create(poly_modulus_degree, bit_sizes);
        }
    };
}
