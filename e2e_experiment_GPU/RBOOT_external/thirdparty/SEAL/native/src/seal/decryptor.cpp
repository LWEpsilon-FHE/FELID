// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/decryptor.h"
#include "seal/valcheck.h"
#include "seal/util/common.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"
#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        void poly_infty_norm_coeffmod(
            StrideIter<const uint64_t *> poly, size_t coeff_count, const uint64_t *modulus, uint64_t *result,
            MemoryPool &pool)
        {
            size_t coeff_uint64_count = poly.stride();

            auto modulus_neg_threshold(allocate_uint(coeff_uint64_count, pool));
            half_round_up_uint(modulus, coeff_uint64_count, modulus_neg_threshold.get());

            set_zero_uint(coeff_uint64_count, result);
            auto coeff_abs_value(allocate_uint(coeff_uint64_count, pool));
            SEAL_ITERATE(poly, coeff_count, [&](auto I) {
                if (is_greater_than_or_equal_uint(I, modulus_neg_threshold.get(), coeff_uint64_count))
                {
                    sub_uint(modulus, I, coeff_uint64_count, coeff_abs_value.get());
                }
                else
                {
                    set_uint(I, coeff_uint64_count, coeff_abs_value.get());
                }

                if (is_greater_than_uint(coeff_abs_value.get(), result, coeff_uint64_count))
                {
                     
                    set_uint(coeff_abs_value.get(), coeff_uint64_count, result);
                }
            });
        }
    }  

    Decryptor::Decryptor(const SEALContext &context, const SecretKey &secret_key) : context_(context)
    {
         
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        if (!is_valid_for(secret_key, context_))
        {
            throw invalid_argument("secret key is not valid for encryption parameters");
        }

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        secret_key_array_ = allocate_poly(coeff_count, coeff_modulus_size, pool_);
        set_poly(secret_key.data().data(), coeff_count, coeff_modulus_size, secret_key_array_.get());
        secret_key_array_size_ = 1;
    }

    void Decryptor::decrypt(const Ciphertext &encrypted, Plaintext &destination)
    {
         
        if (!is_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        if (encrypted.size() < SEAL_CIPHERTEXT_SIZE_MIN)
        {
            throw invalid_argument("encrypted is empty");
        }

        auto &context_data = *context_.first_context_data();
        auto &parms = context_data.parms();

        switch (parms.scheme())
        {
        case scheme_type::bfv:
            bfv_decrypt(encrypted, destination, pool_);
            return;

        case scheme_type::ckks:
            ckks_decrypt(encrypted, destination, pool_);
            return;

        default:
            throw invalid_argument("unsupported scheme");
        }
    }

    void Decryptor::bfv_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        SEAL_ALLOCATE_ZERO_GET_RNS_ITER(tmp_dest_modq, coeff_count, coeff_modulus_size, pool);

        dot_product_ct_sk_array(encrypted, tmp_dest_modq, pool_);

        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        context_data.rns_tool()->decrypt_scale_and_round(tmp_dest_modq, destination.data(), pool);

        size_t plain_coeff_count = get_significant_uint64_count_uint(destination.data(), coeff_count);

        destination.resize(max(plain_coeff_count, size_t(1)));
    }

    void Decryptor::ckks_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t rns_poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);

        destination.parms_id() = parms_id_zero;

        destination.resize(rns_poly_uint64_count);

        dot_product_ct_sk_array(encrypted, RNSIter(destination.data(), coeff_count), pool);

        destination.parms_id() = encrypted.parms_id();
        destination.scale() = encrypted.scale();
    }

    void Decryptor::compute_secret_key_array(size_t max_power)
    {
#ifdef SEAL_DEBUG
        if (max_power < 1)
        {
            throw invalid_argument("max_power must be at least 1");
        }
        if (!secret_key_array_size_ || !secret_key_array_)
        {
            throw logic_error("secret_key_array_ is uninitialized");
        }
#endif
         
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        ReaderLock reader_lock(secret_key_array_locker_.acquire_read());

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);

        if (old_size == new_size)
        {
            return;
        }

        reader_lock.unlock();

        auto secret_key_array(allocate_poly_array(new_size, coeff_count, coeff_modulus_size, pool_));
        PolyIter secret_key_array_iter(secret_key_array.get(), coeff_count, coeff_modulus_size);
        set_poly_array(secret_key_array_.get(), old_size, coeff_count, coeff_modulus_size, secret_key_array_iter);

        SEAL_ITERATE(
            iter(secret_key_array_iter + (old_size - 1), secret_key_array_iter + old_size), new_size - old_size,
            [&](auto I) {
                dyadic_product_coeffmod(
                    get<0>(I), *secret_key_array_iter, coeff_modulus_size, coeff_modulus, get<1>(I));
            });

        WriterLock writer_lock(secret_key_array_locker_.acquire_write());

        old_size = secret_key_array_size_;
        new_size = max(max_power, secret_key_array_size_);

        if (old_size == new_size)
        {
            return;
        }

        secret_key_array_size_ = new_size;
        secret_key_array_.acquire(move(secret_key_array));
    }

    void Decryptor::dot_product_ct_sk_array(const Ciphertext &encrypted, RNSIter destination, MemoryPoolHandle pool)
    {
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t key_coeff_modulus_size = context_.key_context_data()->parms().coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        auto is_ntt_form = encrypted.is_ntt_form();

        auto ntt_tables = context_data.small_ntt_tables();

        compute_secret_key_array(encrypted_size - 1);

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted_copy, encrypted_size - 1, coeff_count, coeff_modulus_size, pool);
        set_poly_array(encrypted.data(1), encrypted_size - 1, coeff_count, coeff_modulus_size, encrypted_copy);

        if (!is_ntt_form)
        {
            ntt_negacyclic_harvey_lazy(encrypted_copy, encrypted_size - 1, ntt_tables);
        }

        auto secret_key_array = PolyIter(secret_key_array_.get(), coeff_count, key_coeff_modulus_size);
        SEAL_ITERATE(iter(encrypted_copy, secret_key_array), encrypted_size - 1, [&](auto I) {
            dyadic_product_coeffmod(get<0>(I), get<1>(I), coeff_modulus_size, coeff_modulus, get<0>(I));
        });

        set_zero_poly(coeff_count, coeff_modulus_size, destination);
        SEAL_ITERATE(encrypted_copy, encrypted_size - 1, [&](auto I) {
            add_poly_coeffmod(destination, I, coeff_modulus_size, coeff_modulus, destination);
        });

        if (!is_ntt_form)
        {
             
            inverse_ntt_negacyclic_harvey(destination, coeff_modulus_size, ntt_tables);
        }

        add_poly_coeffmod(destination, *iter(encrypted), coeff_modulus_size, coeff_modulus, destination);
    }

    int Decryptor::invariant_noise_budget(const Ciphertext &encrypted)
    {
         
        if (!is_valid_for(encrypted, context_))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        if (encrypted.size() < SEAL_CIPHERTEXT_SIZE_MIN)
        {
            throw invalid_argument("encrypted is empty");
        }

        if (context_.key_context_data()->parms().scheme() != scheme_type::bfv)
        {
            throw logic_error("unsupported scheme");
        }
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        auto norm(allocate_uint(coeff_modulus_size, pool_));

        SEAL_ALLOCATE_ZERO_GET_RNS_ITER(noise_poly, coeff_count, coeff_modulus_size, pool_);

        dot_product_ct_sk_array(encrypted, noise_poly, pool_);

        multiply_poly_scalar_coeffmod(noise_poly, coeff_modulus_size, plain_modulus.value(), coeff_modulus, noise_poly);

        context_data.rns_tool()->base_q()->compose_array(noise_poly, coeff_count, pool_);

        StrideIter<const uint64_t *> wide_noise_poly((*noise_poly).ptr(), coeff_modulus_size);
        poly_infty_norm_coeffmod(wide_noise_poly, coeff_count, context_data.total_coeff_modulus(), norm.get(), pool_);

        int bit_count_diff = context_data.total_coeff_modulus_bit_count() -
                             get_significant_bit_count_uint(norm.get(), coeff_modulus_size) - 1;
        return max(0, bit_count_diff);
    }
}
