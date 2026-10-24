// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/keygenerator.h"
#include "seal/randomtostd.h"
#include "seal/util/common.h"
#include "seal/util/galois.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/rlwe.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <algorithm>

using namespace std;
using namespace seal::util;

namespace seal
{
    KeyGenerator::KeyGenerator(const SEALContext &context) : context_(context)
    {
         
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        sk_generated_ = false;

        generate_sk();
    }

    KeyGenerator::KeyGenerator(const SEALContext &context, const SecretKey &secret_key) : context_(context)
    {
         
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        if (!is_valid_for(secret_key, context_))
        {
            throw invalid_argument("secret key is not valid for encryption parameters");
        }

        secret_key_ = secret_key;
        sk_generated_ = true;

        generate_sk(sk_generated_);
    }

    void KeyGenerator::generate_sk(bool is_initialized)
    {
         
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t hamming_weight = parms.secret_key_hamming_weight();  

        if (!is_initialized)
        {
             
            secret_key_ = SecretKey();
            sk_generated_ = false;
            secret_key_.data().resize(mul_safe(coeff_count, coeff_modulus_size));

            RNSIter secret_key(secret_key_.data().data(), coeff_count);
            if(!hamming_weight) sample_poly_ternary(parms.random_generator()->create(), parms, secret_key);  
            else sample_poly_sparse_ternary(parms.random_generator()->create(), parms, secret_key);  

            auto ntt_tables = context_data.small_ntt_tables();
            ntt_negacyclic_harvey(secret_key, coeff_modulus_size, ntt_tables);

            secret_key_.parms_id() = context_data.parms_id();
        }

        secret_key_array_ = allocate_poly(coeff_count, coeff_modulus_size, pool_);
        set_poly(secret_key_.data().data(), coeff_count, coeff_modulus_size, secret_key_array_.get());
        secret_key_array_size_ = 1;

        sk_generated_ = true;
    }

    PublicKey KeyGenerator::generate_pk(bool save_seed) const
    {
        if (!sk_generated_)
        {
            throw logic_error("cannot generate public key for unspecified secret key");
        }

        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        PublicKey public_key;
        encrypt_zero_symmetric(secret_key_, context_, context_data.parms_id(), true, save_seed, public_key.data());

        public_key.parms_id() = context_data.parms_id();

        return public_key;
    }

    RelinKeys KeyGenerator::create_relin_keys(size_t count, bool save_seed)
    {
         
        if (!sk_generated_)
        {
            throw logic_error("cannot generate relinearization keys for unspecified secret key");
        }
        if (!count || count > SEAL_CIPHERTEXT_SIZE_MAX - 2)
        {
            throw invalid_argument("invalid count");
        }

        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        compute_secret_key_array(context_data, count + 1);

        RelinKeys relin_keys;

        ConstPolyIter secret_key(secret_key_array_.get(), coeff_count, coeff_modulus_size);
        generate_kswitch_keys(secret_key + 1, count, static_cast<KSwitchKeys &>(relin_keys), save_seed);

        relin_keys.parms_id() = context_data.parms_id();

        return relin_keys;
    }

    GaloisKeys KeyGenerator::create_galois_keys(const vector<uint32_t> &galois_elts, bool save_seed)
    {

        if (!sk_generated_)
        {
            throw logic_error("cannot generate Galois keys for unspecified secret key");
        }

        auto &context_data = *context_.key_context_data();
        if (!context_data.qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }

        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto galois_tool = context_data.galois_tool();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        GaloisKeys galois_keys;

        galois_keys.data().resize(coeff_count);

		#pragma omp parallel for num_threads(80)	// modified for fast key generation by E.-S. Lee
		for(size_t i=0; i<galois_elts.size(); i++)	 
        {
			uint32_t galois_elt = galois_elts[i];	 

            if (!(galois_elt & 1) || (galois_elt >= coeff_count << 1))
            {
                throw invalid_argument("Galois element is not valid");
            }

            if (galois_keys.has_key(galois_elt))
            {
                continue;
            }

            SEAL_ALLOCATE_GET_RNS_ITER(rotated_secret_key, coeff_count, coeff_modulus_size, pool_);
            RNSIter secret_key(secret_key_.data().data(), coeff_count);
            galois_tool->apply_galois_ntt(secret_key, coeff_modulus_size, galois_elt, rotated_secret_key);

            size_t index = GaloisKeys::get_index(galois_elt);

            generate_one_kswitch_key(rotated_secret_key, galois_keys.data()[index], save_seed);
        }

        galois_keys.parms_id_ = context_data.parms_id();

        return galois_keys;
    }

    GaloisKeys KeyGenerator::create_hoisted_galois_keys(const vector<uint32_t> &galois_elts, const vector<uint32_t> &inv_galois_elts, bool save_seed)
    {
         
        if (!sk_generated_)
        {
            throw logic_error("cannot generate Galois keys for unspecified secret key");
        }

        auto &context_data = *context_.key_context_data();
        if (!context_data.qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }

        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto galois_tool = context_data.galois_tool();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        GaloisKeys galois_keys;

        galois_keys.data().resize(coeff_count);

		#pragma omp parallel for num_threads(80)	// modified for fast key generation by CCL
        for (int i = 0; i < galois_elts.size(); i++) 
        {
			auto galois_elt = galois_elts[i];
			auto inv_galois_elt = inv_galois_elts[i];
             
            if (!(galois_elt & 1) || (galois_elt >= coeff_count << 1))
            {
                throw invalid_argument("Galois element is not valid");
            }

            if (galois_keys.has_key(galois_elt))
            {
                continue;
            }

            SEAL_ALLOCATE_GET_RNS_ITER(inverse_rotated_secret_key, coeff_count, coeff_modulus_size, pool_);
            RNSIter secret_key(secret_key_.data().data(), coeff_count);

            size_t index = GaloisKeys::get_index(galois_elt);
            galois_tool->apply_galois_ntt(secret_key, coeff_modulus_size, inv_galois_elt, inverse_rotated_secret_key);

            generate_one_hoisted_kswitch_key(inverse_rotated_secret_key, galois_keys.data()[index], save_seed);
        }

        galois_keys.parms_id() = context_data.parms_id();

        return galois_keys;
    }

    const SecretKey &KeyGenerator::secret_key() const
    {
        if (!sk_generated_)
        {
            throw logic_error("secret key has not been generated");
        }
        return secret_key_;
    }

    void KeyGenerator::compute_secret_key_array(const SEALContext::ContextData &context_data, size_t max_power)
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
         
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size, max_power))
        {
            throw logic_error("invalid parameters");
        }

        ReaderLock reader_lock(secret_key_array_locker_.acquire_read());

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);

        if (old_size == new_size)
        {
            return;
        }

        reader_lock.unlock();

        auto secret_key_array(allocate_poly_array(new_size, coeff_count, coeff_modulus_size, pool_));
        set_poly_array(secret_key_array_.get(), old_size, coeff_count, coeff_modulus_size, secret_key_array.get());
        RNSIter secret_key(secret_key_array.get(), coeff_count);

        PolyIter secret_key_power(secret_key_array.get(), coeff_count, coeff_modulus_size);
        secret_key_power += (old_size - 1);
        auto next_secret_key_power = secret_key_power + 1;

        SEAL_ITERATE(iter(secret_key_power, next_secret_key_power), new_size - old_size, [&](auto I) {
            dyadic_product_coeffmod(get<0>(I), secret_key, coeff_modulus_size, coeff_modulus, get<1>(I));
        });

        WriterLock writer_lock(secret_key_array_locker_.acquire_write());

        old_size = secret_key_array_size_;
        new_size = max(max_power, secret_key_array_size_);

        if (old_size == new_size)
        {
            return;
        }

        secret_key_array_size_ = new_size;
        secret_key_array_.acquire(secret_key_array);
    }

    void KeyGenerator::generate_one_kswitch_key(ConstRNSIter new_key, vector<PublicKey> &destination, bool save_seed)
    {
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();        
        size_t n_ct_rns = context_.first_context_data()->parms().coeff_modulus().size();
        size_t n_sp_rns = context_.first_context_data()->parms().n_special_primes();
        size_t decomp_mod_count = (n_ct_rns + n_sp_rns - 1) / n_sp_rns;
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto &key_modulus = key_parms.coeff_modulus();

        if (!product_fits_in(coeff_count, decomp_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        destination.resize(decomp_mod_count);

        SEAL_ITERATE(iter(size_t(0), destination), decomp_mod_count, [&](auto I) {
            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count, pool_);
            encrypt_zero_symmetric(secret_key_, context_, key_context_data.parms_id(), true, save_seed, destination[get<0>(I)].data());
            size_t rns0 = get<0>(I) * n_sp_rns;
            size_t rns1 = std::min(rns0 + n_sp_rns, n_ct_rns);

            for (size_t rns = rns0; rns < rns1; ++rns) {
                uint64_t factor{1};  
                for (size_t k = 0; k < n_sp_rns; ++k) {
                    factor = multiply_uint_mod(factor, key_modulus[n_ct_rns + k].value(), key_modulus[rns]);
                }
                 
                CoeffIter destination_iter = (*iter(get<1>(I).data()))[rns];

                multiply_poly_scalar_coeffmod(*(new_key + rns), coeff_count, factor, key_modulus[rns], temp);
                 
                add_poly_coeffmod(destination_iter, temp, coeff_count, key_modulus[rns], destination_iter);
            }
        });
    }

    void KeyGenerator::generate_one_hoisted_kswitch_key(ConstRNSIter inverse_new_key, vector<PublicKey> &destination, bool save_seed)
    {
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        size_t decomp_mod_count = context_.first_context_data()->parms().coeff_modulus().size();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto &key_modulus = key_parms.coeff_modulus();
		size_t key_coeff_modulus_size = key_modulus.size();

        if (!product_fits_in(coeff_count, decomp_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        destination.resize(decomp_mod_count);

		SecretKey inverse_secret_key;
		inverse_secret_key.data().resize(mul_safe(coeff_count, key_coeff_modulus_size));
		RNSIter inverse_secret_key_iter(inverse_secret_key.data().data(), coeff_count);
		set_uint(inverse_new_key, key_coeff_modulus_size * coeff_count, inverse_secret_key_iter);
		inverse_secret_key.parms_id() = key_context_data.parms_id();

		RNSIter secret_key_iter(secret_key_.data().data(), coeff_count);

        SEAL_ITERATE(iter(secret_key_iter, key_modulus, destination, size_t(0)), decomp_mod_count, [&](auto I) {  
            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count, pool_);
            encrypt_zero_symmetric(
                inverse_secret_key, context_, key_context_data.parms_id(), true, save_seed, get<2>(I).data());  
            uint64_t factor = barrett_reduce_64(key_modulus.back().value(), get<1>(I));
            multiply_poly_scalar_coeffmod(get<0>(I), coeff_count, factor, get<1>(I), temp);

            CoeffIter destination_iter = (*iter(get<2>(I).data()))[get<3>(I)];
            add_poly_coeffmod(destination_iter, temp, coeff_count, get<1>(I), destination_iter);
        });
    }

    void KeyGenerator::generate_kswitch_keys(
        ConstPolyIter new_keys, size_t num_keys, KSwitchKeys &destination, bool save_seed)
    {
        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        size_t coeff_modulus_size = key_parms.coeff_modulus().size();

        if (!product_fits_in(coeff_count, coeff_modulus_size, num_keys))
        {
            throw logic_error("invalid parameters");
        }
#ifdef SEAL_DEBUG
        if (new_keys.poly_modulus_degree() != coeff_count)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
        if (new_keys.coeff_modulus_size() != coeff_modulus_size)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
#endif
        destination.data().resize(num_keys);
        SEAL_ITERATE(iter(new_keys, destination.data()), num_keys, [&](auto I) {
            this->generate_one_kswitch_key(get<0>(I), get<1>(I), save_seed);
        });
    }

    void KeyGenerator::generate_hoisted_kswitch_keys(
        ConstPolyIter inverse_new_keys, size_t num_keys, KSwitchKeys &destination, bool save_seed)
    {
        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        size_t coeff_modulus_size = key_parms.coeff_modulus().size();

        if (!product_fits_in(coeff_count, coeff_modulus_size, num_keys))
        {
            throw logic_error("invalid parameters");
        }
#ifdef SEAL_DEBUG
        if (new_keys.poly_modulus_degree() != coeff_count)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
        if (new_keys.coeff_modulus_size() != coeff_modulus_size)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
#endif
        destination.data().resize(num_keys);
        SEAL_ITERATE(iter(inverse_new_keys, destination.data()), num_keys, [&](auto I) {
            this->generate_one_hoisted_kswitch_key(get<0>(I), get<1>(I), save_seed);
        });
    }
}
