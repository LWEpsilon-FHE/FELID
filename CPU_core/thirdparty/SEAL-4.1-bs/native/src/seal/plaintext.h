// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.h"
#include "seal/dynarray.h"
#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/valcheck.h"
#include "seal/version.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include "seal/util/polycore.h"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#ifdef SEAL_USE_MSGSL
#include "gsl/span"
#endif

namespace seal
{
     
    class Plaintext
    {
    public:
        using pt_coeff_type = std::uint64_t;

        Plaintext(MemoryPoolHandle pool = MemoryManager::GetPool()) : data_(std::move(pool))
        {}

        explicit Plaintext(std::size_t coeff_count, MemoryPoolHandle pool = MemoryManager::GetPool())
            : coeff_count_(coeff_count), data_(coeff_count_, std::move(pool))
        {}

        explicit Plaintext(
            std::size_t capacity, std::size_t coeff_count, MemoryPoolHandle pool = MemoryManager::GetPool())
            : coeff_count_(coeff_count), data_(capacity, coeff_count_, std::move(pool))
        {}
#ifdef SEAL_USE_MSGSL
         
        explicit Plaintext(
            gsl::span<const pt_coeff_type> coeffs, std::size_t capacity,
            MemoryPoolHandle pool = MemoryManager::GetPool())
            : coeff_count_(coeffs.size()), data_(coeffs, capacity, std::move(pool))
        {}

        explicit Plaintext(gsl::span<const pt_coeff_type> coeffs, MemoryPoolHandle pool = MemoryManager::GetPool())
            : coeff_count_(coeffs.size()), data_(coeffs, std::move(pool))
        {}
#endif
         
        Plaintext(const std::string &hex_poly, MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
            operator=(hex_poly);
        }

        Plaintext(const Plaintext &copy) = default;

        Plaintext(Plaintext &&source) = default;

        Plaintext(const Plaintext &copy, MemoryPoolHandle pool) : Plaintext(std::move(pool))
        {
            *this = copy;
        }

        void reserve(std::size_t capacity)
        {
            if (is_ntt_form())
            {
                throw std::logic_error("cannot reserve for an NTT transformed Plaintext");
            }
            data_.reserve(capacity);
            coeff_count_ = data_.size();
        }

        inline void shrink_to_fit()
        {
            data_.shrink_to_fit();
        }

        inline void release() noexcept
        {
            parms_id_ = parms_id_zero;
            coeff_count_ = 0;
            scale_ = 1.0;
            data_.release();
        }

        inline void resize(std::size_t coeff_count)
        {
            if (is_ntt_form())
            {
                throw std::logic_error("cannot reserve for an NTT transformed Plaintext");
            }
            data_.resize(coeff_count);
            coeff_count_ = coeff_count;
        }

        Plaintext &operator=(const Plaintext &assign) = default;

        Plaintext &operator=(Plaintext &&assign) = default;

        Plaintext &operator=(const std::string &hex_poly);

        Plaintext &operator=(pt_coeff_type const_coeff)
        {
            data_.resize(1);
            data_[0] = const_coeff;
            coeff_count_ = 1;
            parms_id_ = parms_id_zero;
            return *this;
        }
#ifdef SEAL_USE_MSGSL
         
        Plaintext &operator=(gsl::span<const pt_coeff_type> coeffs)
        {
            data_ = coeffs;
            coeff_count_ = coeffs.size();
            parms_id_ = parms_id_zero;
            return *this;
        }
#endif
         
        inline void set_zero(std::size_t start_coeff, std::size_t length)
        {
            if (!length)
            {
                return;
            }
            if (start_coeff + length - 1 >= coeff_count_)
            {
                throw std::out_of_range(
                    "length must be non-negative and start_coeff + length - 1 must be within [0, coeff_count)");
            }
            std::fill_n(data_.begin() + start_coeff, length, pt_coeff_type(0));
        }

        inline void set_zero(std::size_t start_coeff)
        {
            if (start_coeff >= coeff_count_)
            {
                throw std::out_of_range("start_coeff must be within [0, coeff_count)");
            }
            std::fill(data_.begin() + start_coeff, data_.end(), pt_coeff_type(0));
        }

        inline void set_zero()
        {
            std::fill(data_.begin(), data_.end(), pt_coeff_type(0));
        }

        SEAL_NODISCARD inline const auto &dyn_array() const noexcept
        {
            return data_;
        }

        SEAL_NODISCARD inline pt_coeff_type *data()
        {
            return data_.begin();
        }

        SEAL_NODISCARD inline const pt_coeff_type *data() const
        {
            return data_.cbegin();
        }
#ifdef SEAL_USE_MSGSL
         
        SEAL_NODISCARD inline gsl::span<pt_coeff_type> data_span()
        {
            return gsl::span<pt_coeff_type>(data_.begin(), coeff_count_);
        }

        SEAL_NODISCARD inline gsl::span<const pt_coeff_type> data_span() const
        {
            return gsl::span<const pt_coeff_type>(data_.cbegin(), coeff_count_);
        }
#endif
         
        SEAL_NODISCARD inline pt_coeff_type *data(std::size_t coeff_index)
        {
            if (!coeff_count_)
            {
                return nullptr;
            }
            if (coeff_index >= coeff_count_)
            {
                throw std::out_of_range("coeff_index must be within [0, coeff_count)");
            }
            return data_.begin() + coeff_index;
        }

        SEAL_NODISCARD inline const pt_coeff_type *data(std::size_t coeff_index) const
        {
            if (!coeff_count_)
            {
                return nullptr;
            }
            if (coeff_index >= coeff_count_)
            {
                throw std::out_of_range("coeff_index must be within [0, coeff_count)");
            }
            return data_.cbegin() + coeff_index;
        }

        SEAL_NODISCARD inline const pt_coeff_type &operator[](std::size_t coeff_index) const
        {
            return data_.at(coeff_index);
        }

        SEAL_NODISCARD inline pt_coeff_type &operator[](std::size_t coeff_index)
        {
            return data_.at(coeff_index);
        }

        SEAL_NODISCARD inline bool operator==(const Plaintext &compare) const
        {
            std::size_t sig_coeff_count = significant_coeff_count();
            std::size_t sig_coeff_count_compare = compare.significant_coeff_count();
            bool parms_id_compare = (is_ntt_form() && compare.is_ntt_form() && (parms_id_ == compare.parms_id_)) ||
                                    (!is_ntt_form() && !compare.is_ntt_form());
            return parms_id_compare && (sig_coeff_count == sig_coeff_count_compare) &&
                   std::equal(
                       data_.cbegin(), data_.cbegin() + sig_coeff_count, compare.data_.cbegin(),
                       compare.data_.cbegin() + sig_coeff_count) &&
                   std::all_of(data_.cbegin() + sig_coeff_count, data_.cend(), util::is_zero<pt_coeff_type>) &&
                   std::all_of(
                       compare.data_.cbegin() + sig_coeff_count, compare.data_.cend(), util::is_zero<pt_coeff_type>) &&
                   util::are_close(scale_, compare.scale_);
        }

        SEAL_NODISCARD inline bool operator!=(const Plaintext &compare) const
        {
            return !operator==(compare);
        }

        SEAL_NODISCARD inline bool is_zero() const
        {
            return !coeff_count_ || std::all_of(data_.cbegin(), data_.cend(), util::is_zero<pt_coeff_type>);
        }

        SEAL_NODISCARD inline std::size_t capacity() const noexcept
        {
            return data_.capacity();
        }

        SEAL_NODISCARD inline std::size_t coeff_count() const noexcept
        {
            return coeff_count_;
        }

        SEAL_NODISCARD inline std::size_t significant_coeff_count() const
        {
            if (!coeff_count_)
            {
                return 0;
            }
            return util::get_significant_uint64_count_uint(data_.cbegin(), coeff_count_);
        }

        SEAL_NODISCARD inline std::size_t nonzero_coeff_count() const
        {
            if (!coeff_count_)
            {
                return 0;
            }
            return util::get_nonzero_uint64_count_uint(data_.cbegin(), coeff_count_);
        }

        SEAL_NODISCARD inline std::string to_string() const
        {
            if (is_ntt_form())
            {
                throw std::invalid_argument("cannot convert NTT transformed plaintext to string");
            }
            return util::poly_to_hex_string(data_.cbegin(), coeff_count_, 1);
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            std::size_t members_size = Serialization::ComprSizeEstimate(
                util::add_safe(
                    sizeof(parms_id_),
                    sizeof(std::uint64_t),  
                    sizeof(scale_), util::safe_cast<std::size_t>(data_.save_size(compr_mode_type::none))),
                compr_mode);

            return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Plaintext::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Plaintext::load_members, this, context, _1, _2), stream, false);
        }

        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            Plaintext new_data(pool());
            auto in_size = new_data.unsafe_load(context, stream);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("Plaintext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Plaintext::save_members, this, _1), save_size(compr_mode_type::none), out, size, compr_mode,
                false);
        }

        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Plaintext::load_members, this, context, _1, _2), in, size, false);
        }

        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            Plaintext new_data(pool());
            auto in_size = new_data.unsafe_load(context, in, size);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("Plaintext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        SEAL_NODISCARD inline bool is_ntt_form() const noexcept
        {
            return (parms_id_ != parms_id_zero);
        }

        SEAL_NODISCARD inline parms_id_type &parms_id() noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline const parms_id_type &parms_id() const noexcept
        {
            return parms_id_;
        }

        SEAL_NODISCARD inline double &scale() noexcept
        {
            return scale_;
        }

        SEAL_NODISCARD inline const double &scale() const noexcept
        {
            return scale_;
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return data_.pool();
        }

        struct PlaintextPrivateHelper;

    private:
        void save_members(std::ostream &stream) const;

        void load_members(const SEALContext &context, std::istream &stream, SEALVersion version);

        parms_id_type parms_id_ = parms_id_zero;

        std::size_t coeff_count_ = 0;

        double scale_ = 1.0;

        DynArray<pt_coeff_type> data_;

        friend class SecretKey;
    };
}
