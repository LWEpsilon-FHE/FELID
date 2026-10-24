// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.h"
#include "seal/dynarray.h"
#include "seal/memorymanager.h"
#include "seal/randomgen.h"
#include "seal/valcheck.h"
#include "seal/version.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace seal
{
     
    class Ciphertext
    {
    public:
        using ct_coeff_type = std::uint64_t;

        Ciphertext(MemoryPoolHandle pool = MemoryManager::GetPool()) : data_(std::move(pool))
        {}

        explicit Ciphertext(const SEALContext &context, MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
             
            reserve(context, 2);
        }

        explicit Ciphertext(
            const SEALContext &context, parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
             
            reserve(context, parms_id, 2);
        }

        explicit Ciphertext(
            const SEALContext &context, parms_id_type parms_id, std::size_t size_capacity,
            MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
             
            reserve(context, parms_id, size_capacity);
        }

        Ciphertext(const Ciphertext &copy) = default;

        Ciphertext(Ciphertext &&source) = default;

        Ciphertext(const Ciphertext &copy, MemoryPoolHandle pool) : Ciphertext(std::move(pool))
        {
            *this = copy;
        }

        void reserve(const SEALContext &context, parms_id_type parms_id, std::size_t size_capacity);

        inline void reserve(const SEALContext &context, std::size_t size_capacity)
        {
            auto parms_id = context.first_parms_id();
            reserve(context, parms_id, size_capacity);
        }

        inline void reserve(std::size_t size_capacity)
        {

            reserve_internal(size_capacity, poly_modulus_degree_, coeff_modulus_size_);
        }

        void resize(const SEALContext &context, parms_id_type parms_id, std::size_t size);

        inline void resize(const SEALContext &context, std::size_t size)
        {
            auto parms_id = context.first_parms_id();
            resize(context, parms_id, size);
        }

        inline void resize(std::size_t size)
        {

            resize_internal(size, poly_modulus_degree_, coeff_modulus_size_);
        }

        inline void release() noexcept
        {
            parms_id_ = parms_id_zero;
            is_ntt_form_ = false;
            size_ = 0;
            poly_modulus_degree_ = 0;
            coeff_modulus_size_ = 0;
            scale_ = 1.0;
            data_.release();
        }

        Ciphertext &operator=(const Ciphertext &assign);

        Ciphertext &operator=(Ciphertext &&assign) = default;

        SEAL_NODISCARD inline const auto &dyn_array() const noexcept
        {
            return data_;
        }

        SEAL_NODISCARD inline ct_coeff_type *data() noexcept
        {
            return data_.begin();
        }

        SEAL_NODISCARD inline const ct_coeff_type *data() const noexcept
        {
            return data_.cbegin();
        }

        SEAL_NODISCARD inline ct_coeff_type *data(std::size_t poly_index)
        {
            auto poly_uint64_count = util::mul_safe(poly_modulus_degree_, coeff_modulus_size_);
            if (poly_uint64_count == 0)
            {
                return nullptr;
            }
            if (poly_index >= size_)
            {
                throw std::out_of_range("poly_index must be within [0, size)");
            }
            return data_.begin() + util::safe_cast<std::size_t>(util::mul_safe(poly_index, poly_uint64_count));
        }

        SEAL_NODISCARD inline const ct_coeff_type *data(std::size_t poly_index) const
        {
            auto poly_uint64_count = util::mul_safe(poly_modulus_degree_, coeff_modulus_size_);
            if (poly_uint64_count == 0)
            {
                return nullptr;
            }
            if (poly_index >= size_)
            {
                throw std::out_of_range("poly_index must be within [0, size)");
            }
            return data_.cbegin() + util::safe_cast<std::size_t>(util::mul_safe(poly_index, poly_uint64_count));
        }

        SEAL_NODISCARD inline ct_coeff_type &operator[](std::size_t coeff_index)
        {
            return data_.at(coeff_index);
        }

        SEAL_NODISCARD inline const ct_coeff_type &operator[](std::size_t coeff_index) const
        {
            return data_.at(coeff_index);
        }

        SEAL_NODISCARD inline std::size_t coeff_modulus_size() const noexcept
        {
            return coeff_modulus_size_;
        }

        SEAL_NODISCARD inline std::size_t poly_modulus_degree() const noexcept
        {
            return poly_modulus_degree_;
        }

        SEAL_NODISCARD inline std::size_t size() const noexcept
        {
            return size_;
        }

        SEAL_NODISCARD inline std::size_t size_capacity() const noexcept
        {
            std::size_t poly_uint64_count = poly_modulus_degree_ * coeff_modulus_size_;
            return poly_uint64_count ? data_.capacity() / poly_uint64_count : std::size_t(0);
        }

        SEAL_NODISCARD inline bool is_transparent() const
        {
            return (
                !data_.size() || (size_ < SEAL_CIPHERTEXT_SIZE_MIN) ||
                std::all_of(data(1), data_.cend(), util::is_zero<ct_coeff_type>));
        }

        SEAL_NODISCARD std::streamoff save_size(compr_mode_type compr_mode = Serialization::compr_mode_default) const;

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Ciphertext::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Ciphertext::load_members, this, context, _1, _2), stream, false);
        }

        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            Ciphertext new_data(pool());
            auto in_size = new_data.unsafe_load(context, stream);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("ciphertext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Ciphertext::save_members, this, _1), save_size(compr_mode_type::none), out, size, compr_mode,
                false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Ciphertext::load_members, this, context, _1, _2), in, size, false);
        }

        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            Ciphertext new_data(pool());
            auto in_size = new_data.unsafe_load(context, in, size);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("ciphertext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        SEAL_NODISCARD inline bool is_ntt_form() const noexcept
        {
            return is_ntt_form_;
        }

        SEAL_NODISCARD inline bool &is_ntt_form() noexcept
        {
            return is_ntt_form_;
        }

        SEAL_NODISCARD inline auto &parms_id() noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline auto &parms_id() const noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline auto &scale() noexcept
        {
            return scale_;
        }

        SEAL_NODISCARD inline auto &scale() const noexcept
        {
            return scale_;
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return data_.pool();
        }

        struct CiphertextPrivateHelper;

    private:
        void reserve_internal(
            std::size_t size_capacity, std::size_t poly_modulus_degree, std::size_t coeff_modulus_size);

        void resize_internal(std::size_t size, std::size_t poly_modulus_degree, std::size_t coeff_modulus_size);

        void expand_seed(const SEALContext &context, const UniformRandomGeneratorInfo &prng_info, SEALVersion version);

        void save_members(std::ostream &stream) const;

        void load_members(const SEALContext &context, std::istream &stream, SEALVersion version);

        inline bool has_seed_marker() const noexcept
        {
            return (data_.size() && (size_ == 2)) ? (data(1)[0] == 0xFFFFFFFFFFFFFFFFULL) : false;
        }

        parms_id_type parms_id_ = parms_id_zero;

        bool is_ntt_form_ = false;

        std::size_t size_ = 0;

        std::size_t poly_modulus_degree_ = 0;

        std::size_t coeff_modulus_size_ = 0;

        double scale_ = 1.0;

        DynArray<ct_coeff_type> data_;
    };
}
