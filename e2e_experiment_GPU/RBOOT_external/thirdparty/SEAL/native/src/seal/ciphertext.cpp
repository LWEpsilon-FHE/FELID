// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/ciphertext.h"
#include "seal/util/defines.h"
#include "seal/util/pointer.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include <algorithm>

using namespace std;
using namespace seal::util;

namespace seal
{
    Ciphertext &Ciphertext::operator=(const Ciphertext &assign)
    {
         
        if (this == &assign)
        {
            return *this;
        }

        parms_id_ = assign.parms_id_;
        is_ntt_form_ = assign.is_ntt_form_;
        scale_ = assign.scale_;

        resize_internal(assign.size_, assign.poly_modulus_degree_, assign.coeff_modulus_size_);

        copy(assign.data_.cbegin(), assign.data_.cend(), data_.begin());

        return *this;
    }

    void Ciphertext::reserve(const SEALContext &context, parms_id_type parms_id, size_t size_capacity)
    {
         
        if (!context.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        auto context_data_ptr = context.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &parms = context_data_ptr->parms();
        parms_id_ = context_data_ptr->parms_id();

        reserve_internal(size_capacity, parms.poly_modulus_degree(), parms.coeff_modulus().size());
    }

    void Ciphertext::reserve_internal(size_t size_capacity, size_t poly_modulus_degree, size_t coeff_modulus_size)
    {
        if (size_capacity < SEAL_CIPHERTEXT_SIZE_MIN || size_capacity > SEAL_CIPHERTEXT_SIZE_MAX)
        {
            throw invalid_argument("invalid size_capacity");
        }

        size_t new_data_capacity = mul_safe(size_capacity, poly_modulus_degree, coeff_modulus_size);
        size_t new_data_size = min<size_t>(new_data_capacity, data_.size());

        data_.reserve(new_data_capacity);
        data_.resize(new_data_size);

        size_ = min<size_t>(size_capacity, size_);
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void Ciphertext::resize(const SEALContext &context, parms_id_type parms_id, size_t size)
    {
         
        if (!context.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        auto context_data_ptr = context.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &parms = context_data_ptr->parms();
        parms_id_ = context_data_ptr->parms_id();

        resize_internal(size, parms.poly_modulus_degree(), parms.coeff_modulus().size());
    }

    void Ciphertext::resize_internal(size_t size, size_t poly_modulus_degree, size_t coeff_modulus_size)
    {
        if ((size < SEAL_CIPHERTEXT_SIZE_MIN && size != 0) || size > SEAL_CIPHERTEXT_SIZE_MAX)
        {
            throw invalid_argument("invalid size");
        }

        size_t new_data_size = mul_safe(size, poly_modulus_degree, coeff_modulus_size);
        data_.resize(new_data_size);

        size_ = size;
        poly_modulus_degree_ = poly_modulus_degree;
        coeff_modulus_size_ = coeff_modulus_size;
    }

    void Ciphertext::expand_seed(
        const SEALContext &context, const UniformRandomGeneratorInfo &prng_info, SEALVersion version)
    {
        auto context_data_ptr = context.get_context_data(parms_id_);

        auto prng = prng_info.make_prng();
        if (!prng)
        {
            throw logic_error("unsupported prng_type");
        }

        if (version.major == 3 && version.minor >= 6)
        {
            sample_poly_uniform(prng, context_data_ptr->parms(), data(1));
        }
        else if (version.major == 3 && version.minor == 4)
        {
            sample_poly_uniform_seal_3_4(prng, context_data_ptr->parms(), data(1));
        }
        else if (version.major == 3 && version.minor == 5)
        {
            sample_poly_uniform_seal_3_5(prng, context_data_ptr->parms(), data(1));
        }
        else
        {
             
            throw logic_error("incompatible version");
        }
    }

    streamoff Ciphertext::save_size(compr_mode_type compr_mode) const
    {

        size_t data_size;
        if (has_seed_marker())
        {
             
            DynArray<ct_coeff_type> alias_data(
                Pointer<ct_coeff_type>::Aliasing(const_cast<ct_coeff_type *>(data_.cbegin())), data_.size() / 2, false,
                data_.pool());

            data_size = add_safe(
                safe_cast<size_t>(alias_data.save_size(compr_mode_type::none)),                     
                static_cast<size_t>(UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none)));  
        }
        else
        {
            data_size = safe_cast<size_t>(data_.save_size(compr_mode_type::none));  
        }

        size_t members_size = Serialization::ComprSizeEstimate(
            add_safe(
                sizeof(parms_id_),
                sizeof(seal_byte),  
                sizeof(uint64_t),   
                sizeof(uint64_t),   
                sizeof(uint64_t),   
                sizeof(scale_), data_size),
            compr_mode);

        return safe_cast<streamoff>(add_safe(sizeof(Serialization::SEALHeader), members_size));
    }

    void Ciphertext::save_members(ostream &stream) const
    {
        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(parms_id_type));
            seal_byte is_ntt_form_byte = static_cast<seal_byte>(is_ntt_form_);
            stream.write(reinterpret_cast<const char *>(&is_ntt_form_byte), sizeof(seal_byte));
            uint64_t size64 = safe_cast<uint64_t>(size_);
            stream.write(reinterpret_cast<const char *>(&size64), sizeof(uint64_t));
            uint64_t poly_modulus_degree64 = safe_cast<uint64_t>(poly_modulus_degree_);
            stream.write(reinterpret_cast<const char *>(&poly_modulus_degree64), sizeof(uint64_t));
            uint64_t coeff_modulus_size64 = safe_cast<uint64_t>(coeff_modulus_size_);
            stream.write(reinterpret_cast<const char *>(&coeff_modulus_size64), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));

            if (has_seed_marker())
            {
                UniformRandomGeneratorInfo info;
                size_t info_size = static_cast<size_t>(UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none));
                info.load(reinterpret_cast<const seal_byte *>(data(1) + 1), info_size);

                size_t data_size = data_.size();
                size_t half_size = data_size / 2;

                DynArray<ct_coeff_type> alias_data(data_.pool_);
                alias_data.size_ = half_size;
                alias_data.capacity_ = half_size;
                auto alias_ptr = util::Pointer<ct_coeff_type>::Aliasing(const_cast<ct_coeff_type *>(data_.cbegin()));
                swap(alias_data.data_, alias_ptr);
                alias_data.save(stream, compr_mode_type::none);

                info.save(stream, compr_mode_type::none);
            }
            else
            {
                 
                data_.save(stream, compr_mode_type::none);
            }
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            throw runtime_error("I/O error");
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);
    }

    void Ciphertext::load_members(const SEALContext &context, istream &stream, SEAL_MAYBE_UNUSED SEALVersion version)
    {
         
        if (!context.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        Ciphertext new_data(data_.pool());

        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(parms_id_type));
            seal_byte is_ntt_form_byte;
            stream.read(reinterpret_cast<char *>(&is_ntt_form_byte), sizeof(seal_byte));
            uint64_t size64 = 0;
            stream.read(reinterpret_cast<char *>(&size64), sizeof(uint64_t));
            uint64_t poly_modulus_degree64 = 0;
            stream.read(reinterpret_cast<char *>(&poly_modulus_degree64), sizeof(uint64_t));
            uint64_t coeff_modulus_size64 = 0;
            stream.read(reinterpret_cast<char *>(&coeff_modulus_size64), sizeof(uint64_t));
            double scale = 0;
            stream.read(reinterpret_cast<char *>(&scale), sizeof(double));

            new_data.parms_id_ = parms_id;
            new_data.is_ntt_form_ = (is_ntt_form_byte == seal_byte{}) ? false : true;
            new_data.size_ = safe_cast<size_t>(size64);
            new_data.poly_modulus_degree_ = safe_cast<size_t>(poly_modulus_degree64);
            new_data.coeff_modulus_size_ = safe_cast<size_t>(coeff_modulus_size64);
            new_data.scale_ = scale;

            if (!is_metadata_valid_for(new_data, context, true))
            {
                throw logic_error("ciphertext data is invalid");
            }

            auto total_uint64_count =
                mul_safe(new_data.size_, new_data.poly_modulus_degree_, new_data.coeff_modulus_size_);

            new_data.data_.reserve(total_uint64_count);

            new_data.data_.load(stream, total_uint64_count);

            auto seeded_uint64_count = poly_modulus_degree64 * coeff_modulus_size64;

            if (unsigned_eq(new_data.data_.size(), seeded_uint64_count))
            {

                UniformRandomGeneratorInfo prng_info;

                if (version.major == 3 && version.minor >= 6)
                {
                    prng_info.load(stream);
                }
                else if (version.major == 3 && version.minor >= 4)
                {
                     
                    prng_info.type() = prng_type::blake2xb;
                    stream.read(reinterpret_cast<char *>(&prng_info.seed()), prng_seed_byte_count);
                }
                else
                {
                     
                    throw logic_error("incompatible version");
                }

                new_data.data_.resize(total_uint64_count);
                new_data.expand_seed(context, prng_info, version);
            }

            if (!is_buffer_valid(new_data))
            {
                throw logic_error("ciphertext data is invalid");
            }
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            throw runtime_error("I/O error");
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        swap(*this, new_data);
    }
}
