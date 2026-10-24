// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/encryptionparams.h"
#include "seal/util/uintcore.h"
#include <limits>

using namespace std;
using namespace seal::util;

namespace seal
{
    const parms_id_type parms_id_zero = util::HashFunction::hash_zero_block;

    void EncryptionParameters::save_members(ostream &stream) const
    {
         
        auto old_except_mask = stream.exceptions();
        try
        {
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            uint64_t poly_modulus_degree64 = static_cast<uint64_t>(poly_modulus_degree_);
            uint64_t n_special_primes = static_cast<uint64_t>(n_special_primes_);
            uint64_t coeff_modulus_size64 = static_cast<uint64_t>(coeff_modulus_.size());
            uint8_t scheme = static_cast<uint8_t>(scheme_);

            stream.write(reinterpret_cast<const char *>(&scheme), sizeof(uint8_t));
            stream.write(reinterpret_cast<const char *>(&poly_modulus_degree64), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&n_special_primes), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&coeff_modulus_size64), sizeof(uint64_t));
            for (const auto &mod : coeff_modulus_)
            {
                mod.save(stream, compr_mode_type::none);
            }

            plain_modulus_.save(stream, compr_mode_type::none);
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

    void EncryptionParameters::load_members(istream &stream, SEAL_MAYBE_UNUSED SEALVersion version)
    {
         
        auto old_except_mask = stream.exceptions();
        try
        {
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            uint8_t scheme;
            stream.read(reinterpret_cast<char *>(&scheme), sizeof(uint8_t));

            EncryptionParameters parms(scheme);

            uint64_t poly_modulus_degree64 = 0;
            stream.read(reinterpret_cast<char *>(&poly_modulus_degree64), sizeof(uint64_t));

            if (poly_modulus_degree64 > SEAL_POLY_MOD_DEGREE_MAX)
            {
                throw logic_error("poly_modulus_degree is invalid");
            }

            uint64_t n_special_primes = 0;
            stream.read(reinterpret_cast<char *>(&n_special_primes), sizeof(uint64_t));

            uint64_t coeff_modulus_size64 = 0;
            stream.read(reinterpret_cast<char *>(&coeff_modulus_size64), sizeof(uint64_t));

            if (coeff_modulus_size64 > SEAL_COEFF_MOD_COUNT_MAX)
            {
                throw logic_error("coeff_modulus is invalid");
            }

            vector<Modulus> coeff_modulus;
            for (uint64_t i = 0; i < coeff_modulus_size64; i++)
            {
                coeff_modulus.emplace_back();
                coeff_modulus.back().load(stream);
            }

            if ((n_special_primes > 1 && n_special_primes >= coeff_modulus_size64)
                || n_special_primes < 1)
            {
                throw logic_error("n_special_primes is invalid");
            }

            Modulus plain_modulus;
            plain_modulus.load(stream);

            parms.set_poly_modulus_degree(safe_cast<size_t>(poly_modulus_degree64));
            parms.set_coeff_modulus(coeff_modulus);

            parms.set_plain_modulus(plain_modulus);
            parms.set_n_special_primes(n_special_primes);

            swap(*this, parms);

            stream.exceptions(old_except_mask);
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

    void EncryptionParameters::compute_parms_id()
    {
        size_t coeff_modulus_size = coeff_modulus_.size();

        size_t total_uint64_count = add_safe(
            size_t(1),  
            size_t(1),  
            size_t(1),  
            coeff_modulus_size, plain_modulus_.uint64_count());

        auto param_data(allocate_uint(total_uint64_count, pool_));
        uint64_t *param_data_ptr = param_data.get();

        *param_data_ptr++ = static_cast<uint64_t>(scheme_);

        *param_data_ptr++ = static_cast<uint64_t>(poly_modulus_degree_);
        *param_data_ptr++ = static_cast<uint64_t>(n_special_primes_);

        for (const auto &mod : coeff_modulus_)
        {
            *param_data_ptr++ = mod.value();
        }

        set_uint(plain_modulus_.data(), plain_modulus_.uint64_count(), param_data_ptr);
        param_data_ptr += plain_modulus_.uint64_count();

        HashFunction::hash(param_data.get(), total_uint64_count, parms_id_);

        if (parms_id_ == parms_id_zero)
        {
            throw logic_error("parms_id cannot be zero");
        }
    }
}
