// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/galoiskeys.h"
#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/plaintext.h"
#include "seal/relinkeys.h"
#include "seal/secretkey.h"
#include "seal/valcheck.h"
#include "seal/util/iterator.h"
#include <map>
#include <stdexcept>
#include <vector>

namespace seal
{
     
    class Evaluator
    {
    public:
         
        Evaluator(const SEALContext &context, CKKSEncoder &encoder);

        void negate_inplace(Ciphertext &encrypted)  const;

        inline void negate(const Ciphertext &encrypted, Ciphertext &destination)  const
        {
            destination = encrypted;
            negate_inplace(destination);
        }

        void add_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2)  const;

        inline void add(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination) const
        {
            if (&encrypted2 == &destination)
            {
                add_inplace(destination, encrypted1);
            }
            else
            {
                destination = encrypted1;
                add_inplace(destination, encrypted2);
            }
        }

        void add_many(const std::vector<Ciphertext> &encrypteds, Ciphertext &destination) const;

        void sub_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2) const;

        inline void sub(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination) const
        {
            if (&encrypted2 == &destination)
            {
                sub_inplace(destination, encrypted1);
                negate_inplace(destination);
            }
            else
            {
                destination = encrypted1;
                sub_inplace(destination, encrypted2);
            }
        }

        void multiply_inplace(
            Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void multiply(
            const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (&encrypted2 == &destination)
            {
                multiply_inplace(destination, encrypted1, std::move(pool));
            }
            else
            {
                destination = encrypted1;
                multiply_inplace(destination, encrypted2, std::move(pool));
            }
        }

        void square_inplace(Ciphertext &encrypted, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void square(
            const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            square_inplace(destination, std::move(pool));
        }

        inline void relinearize_inplace(
            Ciphertext &encrypted, const RelinKeys &relin_keys, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            relinearize_internal(encrypted, relin_keys, 2, std::move(pool));
        }

        inline void relinearize(
            const Ciphertext &encrypted, const RelinKeys &relin_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            relinearize_inplace(destination, relin_keys, std::move(pool));
        }

        void mod_switch_to_next(
            const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void mod_switch_to_next_inplace(Ciphertext &encrypted, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            mod_switch_to_next(encrypted, encrypted, std::move(pool));
        }

        inline void mod_switch_to_next_inplace(Plaintext &plain) const
        {
             
            if (!is_valid_for(plain, context_))
            {
                throw std::invalid_argument("plain is not valid for encryption parameters");
            }
            mod_switch_drop_to_next(plain);
        }

        inline void mod_switch_to_next(const Plaintext &plain, Plaintext &destination) const
        {
            destination = plain;
            mod_switch_to_next_inplace(destination);
        }

        void mod_switch_to_inplace(
            Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void mod_switch_to(
            const Ciphertext &encrypted, parms_id_type parms_id, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            mod_switch_to_inplace(destination, parms_id, std::move(pool));
        }

        void mod_switch_to_inplace(Plaintext &plain, parms_id_type parms_id) const;

        inline void mod_switch_to(const Plaintext &plain, parms_id_type parms_id, Plaintext &destination) const
        {
            destination = plain;
            mod_switch_to_inplace(destination, parms_id);
        }

        void rescale_to_next(
            const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void rescale_to_next_inplace(Ciphertext &encrypted, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            rescale_to_next(encrypted, encrypted, std::move(pool));
        }

        void rescale_to_inplace(
            Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void rescale_to(
            const Ciphertext &encrypted, parms_id_type parms_id, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            rescale_to_inplace(destination, parms_id, std::move(pool));
        }

        void multiply_many(
            const std::vector<Ciphertext> &encrypteds, const RelinKeys &relin_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        void exponentiate_inplace(
            Ciphertext &encrypted, std::uint64_t exponent, const RelinKeys &relin_keys,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void exponentiate(
            const Ciphertext &encrypted, std::uint64_t exponent, const RelinKeys &relin_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            exponentiate_inplace(destination, exponent, relin_keys, std::move(pool));
        }

        void add_plain_inplace(Ciphertext &encrypted, const Plaintext &plain) const;

        inline void add_plain(const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination) const
        {
            destination = encrypted;
            add_plain_inplace(destination, plain);
        }

        void sub_plain_inplace(Ciphertext &encrypted, const Plaintext &plain) const;

        inline void sub_plain(const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination) const
        {
            destination = encrypted;
            sub_plain_inplace(destination, plain);
        }

        void multiply_plain_inplace(
            Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void multiply_plain(
            const Ciphertext &encrypted, const Plaintext &plain, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            multiply_plain_inplace(destination, plain, std::move(pool));
        }

        void transform_to_ntt_inplace(
            Plaintext &plain, parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void transform_to_ntt(
            const Plaintext &plain, parms_id_type parms_id, Plaintext &destination_ntt,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination_ntt = plain;
            transform_to_ntt_inplace(destination_ntt, parms_id, std::move(pool));
        }

        void transform_to_ntt_inplace(Ciphertext &encrypted) const;

        inline void transform_to_ntt(const Ciphertext &encrypted, Ciphertext &destination_ntt) const
        {
            destination_ntt = encrypted;
            transform_to_ntt_inplace(destination_ntt);
        }

        void transform_from_ntt_inplace(Ciphertext &encrypted_ntt) const;
        void transform_from_ntt_inplace(Plaintext &decrypted_ntt) const;

        inline void transform_from_ntt(const Ciphertext &encrypted_ntt, Ciphertext &destination) const
        {
            destination = encrypted_ntt;
            transform_from_ntt_inplace(destination);
        }

        void apply_galois_inplace(
            Ciphertext &encrypted, std::uint32_t galois_elt, const GaloisKeys &galois_keys,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        inline void apply_galois(
            const Ciphertext &encrypted, std::uint32_t galois_elt, const GaloisKeys &galois_keys,
            Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            apply_galois_inplace(destination, galois_elt, galois_keys, std::move(pool));
        }

        inline void rotate_rows_inplace(
            Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (context_.key_context_data()->parms().scheme() != scheme_type::bfv)
            {
                throw std::logic_error("unsupported scheme");
            }
            rotate_internal(encrypted, steps, galois_keys, std::move(pool));
        }

        inline void rotate_rows(
            const Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            rotate_rows_inplace(destination, steps, galois_keys, std::move(pool));
        }

        inline void rotate_columns_inplace(
            Ciphertext &encrypted, const GaloisKeys &galois_keys, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (context_.key_context_data()->parms().scheme() != scheme_type::bfv)
            {
                throw std::logic_error("unsupported scheme");
            }
            conjugate_internal(encrypted, galois_keys, std::move(pool));
        }

        inline void rotate_columns(
            const Ciphertext &encrypted, const GaloisKeys &galois_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            rotate_columns_inplace(destination, galois_keys, std::move(pool));
        }

        inline void rotate_vector_inplace(
            Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (context_.key_context_data()->parms().scheme() != scheme_type::ckks)
            {
                throw std::logic_error("unsupported scheme");
            }
            rotate_internal(encrypted, steps, galois_keys, std::move(pool));
        }

        inline void rotate_vector(
            const Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            rotate_vector_inplace(destination, steps, galois_keys, std::move(pool));
        }

        inline void complex_conjugate_inplace(
            Ciphertext &encrypted, const GaloisKeys &galois_keys, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (context_.key_context_data()->parms().scheme() != scheme_type::ckks)
            {
                throw std::logic_error("unsupported scheme");
            }
            conjugate_internal(encrypted, galois_keys, std::move(pool));
        }

        inline void complex_conjugate(
            const Ciphertext &encrypted, const GaloisKeys &galois_keys, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination = encrypted;
            complex_conjugate_inplace(destination, galois_keys, std::move(pool));
        }

        void add_const_inplace(Ciphertext &encrypted, double value) const;

        inline void add_const(const Ciphertext &encrypted, double value, Ciphertext &destination) const{
            destination = encrypted;
            add_const_inplace(destination, value);
        }

        void add_const_inplace(Ciphertext &encrypted, NTL::RR value) const;

        inline void add_const(const Ciphertext &encrypted, NTL::RR value, Ciphertext &destination) const{
            destination = encrypted;
            add_const_inplace(destination, value);
        }

        void multiply_const_inplace(Ciphertext &encrypted, double value) const;

        inline void multiply_const(const Ciphertext &encrypted, double value, Ciphertext &destination) const {
            destination = encrypted;
            multiply_const_inplace(destination, value);
        }

        void multiply_const_inplace(Ciphertext &encrypted, NTL::RR value) const;

        void multiply_const_inplace_fit_to_scale(Ciphertext &encrypted, NTL::RR value, NTL::RR final_scale) const;

        inline void multiply_const(const Ciphertext &encrypted, NTL::RR value, Ciphertext &destination) const {
            destination = encrypted;
            multiply_const_inplace(destination, value);
        }

        template <typename T>
        void multiply_vector_inplace(Ciphertext &encrypted, const std::vector<T> &value) const;

        template <typename T>
        void multiply_vector(Ciphertext &encrypted, const std::vector<T> &value, Ciphertext &destination) const {
            destination = encrypted;
            multiply_vector_inplace(destination, value);
        }

        void add_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2) const;

        inline void add_reduced_error(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination) const
        {
            if (&encrypted2 == &destination)
            {
                add_inplace_reduced_error(destination, encrypted1);
            }
            else
            {
                destination = encrypted1;
                add_inplace_reduced_error(destination, encrypted2);
            }
        }

        inline void double_inplace(Ciphertext & encrypted) const {
            add_inplace(encrypted, encrypted);
        }

        void sub_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2)  const;

        inline void sub_reduced_error(const Ciphertext &encrypted1, const Ciphertext &encrypted2, Ciphertext &destination) const
        {
            if (&encrypted2 == &destination)
            {
                sub_inplace_reduced_error(destination, encrypted1);
            }
            else
            {
                destination = encrypted1;
                sub_inplace_reduced_error(destination, encrypted2);
            }
        }

        void multiply_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2, const RelinKeys &relin_keys) const;

        inline void multiply_reduced_error(const Ciphertext &encrypted1, const Ciphertext &encrypted2, const RelinKeys &relin_keys, Ciphertext &destination) const
        {
            if (&encrypted2 == &destination)
            {
                multiply_inplace_reduced_error(destination, encrypted1, relin_keys);
            }
            else
            {
                destination = encrypted1;
                multiply_inplace_reduced_error(destination, encrypted2, relin_keys);
            }
        }

		template <typename T, typename = std::enable_if_t<std::is_same<std::remove_cv_t<T>, double>::value ||std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
		void multiply_vector_inplace_reduced_error(Ciphertext &encrypted, const std::vector<T> &value)
		{
			Plaintext plain;
			
			encoder_.encode(value, encrypted.parms_id(), encrypted.scale(), plain);
			mod_switch_to_inplace(plain, encrypted.parms_id());
			multiply_plain_inplace(encrypted, plain);
		}

		template <typename T, typename = std::enable_if_t<std::is_same<std::remove_cv_t<T>, double>::value ||std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
		inline void multiply_vector_reduced_error(Ciphertext &encrypted, const std::vector<T> &value, Ciphertext &destination)
		{
			destination = encrypted;
			multiply_vector_inplace_reduced_error(destination, value);
		}

        struct EvaluatorPrivateHelper;

    private:
        Evaluator(const Evaluator &copy) = delete;

        Evaluator(Evaluator &&source) = delete;

        Evaluator &operator=(const Evaluator &assign) = delete;

        Evaluator &operator=(Evaluator &&assign) = delete;

        void bfv_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool)  const;

        void ckks_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool)  const;

        void bfv_square(Ciphertext &encrypted, MemoryPoolHandle pool) const;

        void ckks_square(Ciphertext &encrypted, MemoryPoolHandle pool) const;

        void relinearize_internal(
            Ciphertext &encrypted, const RelinKeys &relin_keys, std::size_t destination_size, MemoryPoolHandle pool) const;

        void mod_switch_scale_to_next(const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const;

        void mod_switch_drop_to_next(const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const;

        void mod_switch_drop_to_next(Plaintext &plain) const;

        void rotate_internal(Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const;

        inline void conjugate_internal(Ciphertext &encrypted, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const
        {
             
            auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
            if (!context_data_ptr)
            {
                throw std::invalid_argument("encrypted is not valid for encryption parameters");
            }

            auto &context_data = *context_data_ptr;
            if (!context_data.qualifiers().using_batching)
            {
                throw std::logic_error("encryption parameters do not support batching");
            }

            auto galois_tool = context_data.galois_tool();

            apply_galois_inplace(encrypted, galois_tool->get_elt_from_step(0), galois_keys, std::move(pool));
        }

        void switch_key_inplace(
            Ciphertext &encrypted, util::ConstRNSIter target_iter, const KSwitchKeys &kswitch_keys,
            std::size_t key_index, MemoryPoolHandle pool = MemoryManager::GetPool())  const;

        void multiply_plain_normal(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const;

        void multiply_plain_ntt(Ciphertext &encrypted_ntt, const Plaintext &plain_ntt) const;

        void populate_Zmstar_to_generator();

        SEALContext context_;

        CKKSEncoder &encoder_;

        std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> Zmstar_to_generator_{};
    };
}
