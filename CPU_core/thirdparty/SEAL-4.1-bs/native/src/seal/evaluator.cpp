// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/evaluator.h"
#include "seal/util/common.h"
#include "seal/util/galois.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.h"
#include <algorithm>
#include <cmath>
#include <functional>

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        template <typename T, typename S>
        SEAL_NODISCARD inline bool are_same_scale(const T &value1, const S &value2) noexcept
        {
            return util::are_close<double>(value1.scale(), value2.scale());
        }

        SEAL_NODISCARD inline bool is_scale_within_bounds(
            double scale, const SEALContext::ContextData &context_data) noexcept
        {
            int scale_bit_count_bound = 0;
            switch (context_data.parms().scheme())
            {
            case scheme_type::bfv:
            case scheme_type::bgv:
                scale_bit_count_bound = context_data.parms().plain_modulus().bit_count();
                break;
            case scheme_type::ckks:
                scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                break;
            default:
                 
                scale_bit_count_bound = -1;
            };

            return !(scale <= 0 || (static_cast<int>(log2(scale)) >= scale_bit_count_bound));
        }

        SEAL_NODISCARD inline auto balance_correction_factors(
            uint64_t factor1, uint64_t factor2, const Modulus &plain_modulus) -> tuple<uint64_t, uint64_t, uint64_t>
        {
            uint64_t t = plain_modulus.value();
            uint64_t half_t = t / 2;

            auto sum_abs = [&](uint64_t x, uint64_t y) {
                int64_t x_bal = static_cast<int64_t>(x > half_t ? x - t : x);
                int64_t y_bal = static_cast<int64_t>(y > half_t ? y - t : y);
                return abs(x_bal) + abs(y_bal);
            };

            uint64_t ratio = 1;
            if (!try_invert_uint_mod(factor1, plain_modulus, ratio))
            {
                throw logic_error("invalid correction factor1");
            }
            ratio = multiply_uint_mod(ratio, factor2, plain_modulus);
            uint64_t e1 = ratio;
            uint64_t e2 = 1;
            int64_t sum = sum_abs(e1, e2);

            int64_t prev_a = static_cast<int64_t>(plain_modulus.value());
            int64_t prev_b = static_cast<int64_t>(0);
            int64_t a = static_cast<int64_t>(ratio);
            int64_t b = 1;

            while (a != 0)
            {
                int64_t q = prev_a / a;
                int64_t temp = prev_a % a;
                prev_a = a;
                a = temp;

                temp = sub_safe(prev_b, mul_safe(b, q));
                prev_b = b;
                b = temp;

                uint64_t a_mod = barrett_reduce_64(static_cast<uint64_t>(abs(a)), plain_modulus);
                if (a < 0)
                {
                    a_mod = negate_uint_mod(a_mod, plain_modulus);
                }
                uint64_t b_mod = barrett_reduce_64(static_cast<uint64_t>(abs(b)), plain_modulus);
                if (b < 0)
                {
                    b_mod = negate_uint_mod(b_mod, plain_modulus);
                }
                if (a_mod != 0 && gcd(a_mod, t) == 1)  
                {
                    int64_t new_sum = sum_abs(a_mod, b_mod);
                    if (new_sum < sum)
                    {
                        sum = new_sum;
                        e1 = a_mod;
                        e2 = b_mod;
                    }
                }
            }
            return make_tuple(multiply_uint_mod(e1, factor1, plain_modulus), e1, e2);
        }
    }  

    Evaluator::Evaluator(const SEALContext &context, CKKSEncoder &encoder) : context_(context), encoder_(encoder)
    {
         
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
    }

    void Evaluator::negate_inplace(Ciphertext &encrypted) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t encrypted_size = encrypted.size();

        negate_poly_coeffmod(encrypted, encrypted_size, coeff_modulus, encrypted);
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::add_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2) const
    {
         
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        if (encrypted1.correction_factor() != encrypted2.correction_factor())
        {
             
            auto factors = balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(), plain_modulus);
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted1.data(), coeff_count, coeff_modulus_size), encrypted1.size(), get<1>(factors),
                coeff_modulus, PolyIter(encrypted1.data(), coeff_count, coeff_modulus_size));

            Ciphertext encrypted2_copy = encrypted2;
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted2.data(), coeff_count, coeff_modulus_size), encrypted2.size(), get<2>(factors),
                coeff_modulus, PolyIter(encrypted2_copy.data(), coeff_count, coeff_modulus_size));

            encrypted1.correction_factor() = get<0>(factors);
            encrypted2_copy.correction_factor() = get<0>(factors);

            add_inplace(encrypted1, encrypted2_copy);
        }
        else
        {
             
            encrypted1.resize(context_, context_data.parms_id(), max_count);
             
            add_poly_coeffmod(encrypted1, encrypted2, min_count, coeff_modulus, encrypted1);

            if (encrypted1_size < encrypted2_size)
            {
                set_poly_array(
                    encrypted2.data(min_count), encrypted2_size - encrypted1_size, coeff_count, coeff_modulus_size,
                    encrypted1.data(encrypted1_size));
            }
        }

#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::add_many(const vector<Ciphertext> &encrypteds, Ciphertext &destination) const
    {
        if (encrypteds.empty())
        {
            throw invalid_argument("encrypteds cannot be empty");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }

        destination = encrypteds[0];
        for (size_t i = 1; i < encrypteds.size(); i++)
        {
            add_inplace(destination, encrypteds[i]);
        }
    }

    void Evaluator::sub_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2) const
    {
         
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        if (encrypted1.correction_factor() != encrypted2.correction_factor())
        {
             
            auto factors = balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(), plain_modulus);

            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted1.data(), coeff_count, coeff_modulus_size), encrypted1.size(), get<1>(factors),
                coeff_modulus, PolyIter(encrypted1.data(), coeff_count, coeff_modulus_size));

            Ciphertext encrypted2_copy = encrypted2;
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted2.data(), coeff_count, coeff_modulus_size), encrypted2.size(), get<2>(factors),
                coeff_modulus, PolyIter(encrypted2_copy.data(), coeff_count, coeff_modulus_size));

            encrypted1.correction_factor() = get<0>(factors);
            encrypted2_copy.correction_factor() = get<0>(factors);

            sub_inplace(encrypted1, encrypted2_copy);
        }
        else
        {
             
            encrypted1.resize(context_, context_data.parms_id(), max_count);

            sub_poly_coeffmod(encrypted1, encrypted2, min_count, coeff_modulus, encrypted1);

            if (encrypted1_size < encrypted2_size)
            {
                negate_poly_coeffmod(
                    iter(encrypted2) + min_count, encrypted2_size - min_count, coeff_modulus,
                    iter(encrypted1) + min_count);
            }
        }

#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }

        auto context_data_ptr = context_.first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::bfv:
            bfv_multiply(encrypted1, encrypted2, pool);
            break;

        case scheme_type::ckks:
            ckks_multiply(encrypted1, encrypted2, pool);
            break;

        case scheme_type::bgv:
            bgv_multiply(encrypted1, encrypted2, pool);
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted1.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::add_const_inplace(Ciphertext &encrypted, double value) const {
        Plaintext const_plain;

        encoder_.encode(value, encrypted.scale(), const_plain);
        mod_switch_to_inplace(const_plain, encrypted.parms_id());
        add_plain_inplace(encrypted, const_plain);
    }

    void Evaluator::multiply_const_inplace(Ciphertext &encrypted, double value) const{
        Plaintext const_plain;

        encoder_.encode(value, encrypted.scale(), const_plain);
        mod_switch_to_inplace(const_plain, encrypted.parms_id());
        multiply_plain_inplace(encrypted, const_plain);
    }

    template <typename T>
    void Evaluator::multiply_vector_inplace(Ciphertext &encrypted, const std::vector<T> &value) const {
        Plaintext vector_plain;

        encoder_.encode(value, encrypted.scale(), vector_plain);
        mod_switch_to_inplace(vector_plain, encrypted.parms_id());
        multiply_plain_inplace(encrypted, vector_plain);
    }

    void Evaluator::add_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2) const {
        size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
        size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

        if(encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size)
        {
            encrypted1.scale() = encrypted2.scale();
            add_inplace(encrypted1, encrypted2);
            return;
        }
        else if(encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size)
        {
            auto &context_data = *context_.get_context_data(encrypted2.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted2_adjusted;

			double scale_adjust = encrypted1.scale()*static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value())/(encrypted2.scale()*encrypted2.scale());
			multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
			encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted2_adjusted);
			mod_switch_to_inplace(encrypted2_adjusted, encrypted1.parms_id());
			encrypted1.scale() = encrypted2_adjusted.scale();	
			add_inplace(encrypted1, encrypted2_adjusted);

        }
        else
        {
            auto &context_data = *context_.get_context_data(encrypted1.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted1_adjusted;

			double scale_adjust = encrypted2.scale()*static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value())/(encrypted1.scale()*encrypted1.scale());
			multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
			encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted1_adjusted);
			mod_switch_to_inplace(encrypted1_adjusted, encrypted2.parms_id());
			encrypted1_adjusted.scale() = encrypted2.scale();	
			add(encrypted1_adjusted, encrypted2, encrypted1);

        }
    }

    void Evaluator::sub_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2) const {
        size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
        size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

        if(encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size)
        {
            encrypted1.scale() = encrypted2.scale();
            sub_inplace(encrypted1, encrypted2);
            return;
        }
        else if(encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size)
        {
            auto &context_data = *context_.get_context_data(encrypted2.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted2_adjusted;

			double scale_adjust = encrypted1.scale()*static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value())/(encrypted2.scale()*encrypted2.scale());
			multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
			encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted2_adjusted);
			mod_switch_to_inplace(encrypted2_adjusted, encrypted1.parms_id());
			encrypted1.scale() = encrypted2_adjusted.scale();	
			sub_inplace(encrypted1, encrypted2_adjusted);

        }
        else
        {
            auto &context_data = *context_.get_context_data(encrypted1.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted1_adjusted;

			double scale_adjust = encrypted2.scale()*static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value())/(encrypted1.scale()*encrypted1.scale());
			multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
			encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted1_adjusted);
			mod_switch_to_inplace(encrypted1_adjusted, encrypted2.parms_id());
			encrypted1_adjusted.scale() = encrypted2.scale();
			sub(encrypted1_adjusted, encrypted2, encrypted1);

        }
    }

    void Evaluator::multiply_inplace_reduced_error(Ciphertext &encrypted1, const Ciphertext &encrypted2, const RelinKeys &relin_keys) const {
        size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
        size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

        if(encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size)
        {
            encrypted1.scale() = encrypted2.scale();
            multiply_inplace(encrypted1, encrypted2);
			relinearize_inplace(encrypted1, relin_keys);
            return;
        }
        else if(encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size)
        {
            auto &context_data = *context_.get_context_data(encrypted2.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted2_adjusted;

			double scale_adjust = encrypted1.scale()*static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value())/(encrypted2.scale()*encrypted2.scale());
			multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
			encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted2_adjusted);
			mod_switch_to_inplace(encrypted2_adjusted, encrypted1.parms_id());
			encrypted1.scale() = encrypted2_adjusted.scale();	
			multiply_inplace(encrypted1, encrypted2_adjusted);

        }
        else
        {
            auto &context_data = *context_.get_context_data(encrypted1.parms_id());
            auto &parms = context_data.parms();
            auto modulus = iter(parms.coeff_modulus());
            Ciphertext encrypted1_adjusted;

			double scale_adjust = encrypted2.scale()*static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value())/(encrypted1.scale()*encrypted1.scale());
			multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
			encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
			rescale_to_next_inplace(encrypted1_adjusted);
			mod_switch_to_inplace(encrypted1_adjusted, encrypted2.parms_id());
			encrypted1_adjusted.scale() = encrypted2.scale();
			multiply(encrypted1_adjusted, encrypted2, encrypted1);

        }
        relinearize_inplace(encrypted1, relin_keys);
    }

    void Evaluator::bfv_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form())
        {
            throw invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t base_q_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        uint64_t plain_modulus = parms.plain_modulus().value();

        auto rns_tool = context_data.rns_tool();
        size_t base_Bsk_size = rns_tool->base_Bsk()->size();
        size_t base_Bsk_m_tilde_size = rns_tool->base_Bsk_m_tilde()->size();

        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        if (!product_fits_in(dest_size, coeff_count, base_Bsk_m_tilde_size))
        {
            throw logic_error("invalid parameters");
        }

        auto base_q = iter(parms.coeff_modulus());
        auto base_Bsk = iter(rns_tool->base_Bsk()->base());

        auto base_q_ntt_tables = iter(context_data.small_ntt_tables());
        auto base_Bsk_ntt_tables = iter(rns_tool->base_Bsk_ntt_tables());

        encrypted1.resize(context_, context_data.parms_id(), dest_size);

        auto behz_extend_base_convert_to_ntt = [&](auto I) {

            set_poly(get<0>(I), coeff_count, base_q_size, get<1>(I));
            ntt_negacyclic_harvey_lazy(get<1>(I), base_q_size, base_q_ntt_tables);

            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count, base_Bsk_m_tilde_size, pool);

            rns_tool->fastbconv_m_tilde(get<0>(I), temp, pool);

            rns_tool->sm_mrq(temp, get<2>(I), pool);

            ntt_negacyclic_harvey_lazy(get<2>(I), base_Bsk_size, base_Bsk_ntt_tables);
        };

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted1_q, encrypted1_size, coeff_count, base_q_size, pool);

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted1_Bsk, encrypted1_size, coeff_count, base_Bsk_size, pool);

        SEAL_ITERATE(iter(encrypted1, encrypted1_q, encrypted1_Bsk), encrypted1_size, behz_extend_base_convert_to_ntt);

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted2_q, encrypted2_size, coeff_count, base_q_size, pool);
        SEAL_ALLOCATE_GET_POLY_ITER(encrypted2_Bsk, encrypted2_size, coeff_count, base_Bsk_size, pool);

        SEAL_ITERATE(iter(encrypted2, encrypted2_q, encrypted2_Bsk), encrypted2_size, behz_extend_base_convert_to_ntt);

        SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp_dest_q, dest_size, coeff_count, base_q_size, pool);
        SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp_dest_Bsk, dest_size, coeff_count, base_Bsk_size, pool);

        SEAL_ITERATE(iter(size_t(0)), dest_size, [&](auto I) {

            size_t curr_encrypted1_last = min<size_t>(I, encrypted1_size - 1);
            size_t curr_encrypted2_first = min<size_t>(I, encrypted2_size - 1);
            size_t curr_encrypted1_first = I - curr_encrypted2_first;

            size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

            auto behz_ciphertext_product = [&](ConstPolyIter in1_iter, ConstPolyIter in2_iter,
                                               ConstModulusIter base_iter, size_t base_size, PolyIter out_iter) {
                 
                auto shifted_in1_iter = in1_iter + curr_encrypted1_first;

                auto shifted_reversed_in2_iter = reverse_iter(in2_iter + curr_encrypted2_first);

                auto shifted_out_iter = out_iter[I];

                SEAL_ITERATE(iter(shifted_in1_iter, shifted_reversed_in2_iter), steps, [&](auto J) {
                    SEAL_ITERATE(iter(J, base_iter, shifted_out_iter), base_size, [&](auto K) {
                        SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count, pool);
                        dyadic_product_coeffmod(get<0, 0>(K), get<0, 1>(K), coeff_count, get<1>(K), temp);
                        add_poly_coeffmod(temp, get<2>(K), coeff_count, get<1>(K), get<2>(K));
                    });
                });
            };

            behz_ciphertext_product(encrypted1_q, encrypted2_q, base_q, base_q_size, temp_dest_q);
            behz_ciphertext_product(encrypted1_Bsk, encrypted2_Bsk, base_Bsk, base_Bsk_size, temp_dest_Bsk);
        });

        inverse_ntt_negacyclic_harvey_lazy(temp_dest_q, dest_size, base_q_ntt_tables);
        inverse_ntt_negacyclic_harvey_lazy(temp_dest_Bsk, dest_size, base_Bsk_ntt_tables);

        SEAL_ITERATE(iter(temp_dest_q, temp_dest_Bsk, encrypted1), dest_size, [&](auto I) {
             
            SEAL_ALLOCATE_GET_RNS_ITER(temp_q_Bsk, coeff_count, base_q_size + base_Bsk_size, pool);

            multiply_poly_scalar_coeffmod(get<0>(I), base_q_size, plain_modulus, base_q, temp_q_Bsk);

            multiply_poly_scalar_coeffmod(get<1>(I), base_Bsk_size, plain_modulus, base_Bsk, temp_q_Bsk + base_q_size);

            SEAL_ALLOCATE_GET_RNS_ITER(temp_Bsk, coeff_count, base_Bsk_size, pool);

            rns_tool->fast_floor(temp_q_Bsk, temp_Bsk, pool);

            rns_tool->fastbconv_sk(temp_Bsk, get<2>(I), pool);
        });
    }

    void Evaluator::ckks_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
        {
            throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        auto coeff_modulus = iter(parms.coeff_modulus());

        encrypted1.resize(context_, context_data.parms_id(), dest_size);

        PolyIter encrypted1_iter = iter(encrypted1);
        ConstPolyIter encrypted2_iter = iter(encrypted2);

        if (dest_size == 3)
        {

            size_t tile_size = min<size_t>(coeff_count, size_t(256));
            size_t num_tiles = coeff_count / tile_size;
#ifdef SEAL_DEBUG
            if (coeff_count % tile_size != 0)
            {
                throw invalid_argument("tile_size does not divide coeff_count");
            }
#endif

            ConstRNSIter encrypted2_0_iter(*encrypted2_iter[0], tile_size);
            ConstRNSIter encrypted2_1_iter(*encrypted2_iter[1], tile_size);
            RNSIter encrypted1_0_iter(*encrypted1_iter[0], tile_size);
            RNSIter encrypted1_1_iter(*encrypted1_iter[1], tile_size);
            RNSIter encrypted1_2_iter(*encrypted1_iter[2], tile_size);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, tile_size, pool);

            SEAL_ITERATE(coeff_modulus, coeff_modulus_size, [&](auto I) {
                SEAL_ITERATE(iter(size_t(0)), num_tiles, [&](SEAL_MAYBE_UNUSED auto J) {

                    dyadic_product_coeffmod(
                        encrypted1_1_iter[0], encrypted2_1_iter[0], tile_size, I, encrypted1_2_iter[0]);

                    dyadic_product_coeffmod(encrypted1_1_iter[0], encrypted2_0_iter[0], tile_size, I, temp);
                     
                    dyadic_product_coeffmod(
                        encrypted1_0_iter[0], encrypted2_1_iter[0], tile_size, I, encrypted1_1_iter[0]);
                     
                    add_poly_coeffmod(encrypted1_1_iter[0], temp, tile_size, I, encrypted1_1_iter[0]);

                    dyadic_product_coeffmod(
                        encrypted1_0_iter[0], encrypted2_0_iter[0], tile_size, I, encrypted1_0_iter[0]);

                    encrypted1_0_iter++;
                    encrypted1_1_iter++;
                    encrypted1_2_iter++;
                    encrypted2_0_iter++;
                    encrypted2_1_iter++;
                });
            });
        }
        else
        {
             
            SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp, dest_size, coeff_count, coeff_modulus_size, pool);

            SEAL_ITERATE(iter(size_t(0)), dest_size, [&](auto I) {

                size_t curr_encrypted1_last = min<size_t>(I, encrypted1_size - 1);
                size_t curr_encrypted2_first = min<size_t>(I, encrypted2_size - 1);
                size_t curr_encrypted1_first = I - curr_encrypted2_first;

                size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                auto shifted_encrypted1_iter = encrypted1_iter + curr_encrypted1_first;

                auto shifted_reversed_encrypted2_iter = reverse_iter(encrypted2_iter + curr_encrypted2_first);

                SEAL_ITERATE(iter(shifted_encrypted1_iter, shifted_reversed_encrypted2_iter), steps, [&](auto J) {

                    SEAL_ITERATE(iter(J, coeff_modulus, temp[I]), coeff_modulus_size, [&](auto K) {
                        SEAL_ALLOCATE_GET_COEFF_ITER(prod, coeff_count, pool);
                        dyadic_product_coeffmod(get<0, 0>(K), get<0, 1>(K), coeff_count, get<1>(K), prod);
                        add_poly_coeffmod(prod, get<2>(K), coeff_count, get<1>(K), get<2>(K));
                    });
                });
            });

            set_poly_array(temp, dest_size, coeff_count, coeff_modulus_size, encrypted1.data());
        }

        encrypted1.scale() *= encrypted2.scale();
        if (!is_scale_within_bounds(encrypted1.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::bgv_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (!encrypted1.is_ntt_form() || !encrypted2.is_ntt_form())
        {
            throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        auto coeff_modulus = iter(parms.coeff_modulus());

        encrypted1.resize(context_, context_data.parms_id(), dest_size);

        PolyIter encrypted1_iter = iter(encrypted1);
        ConstPolyIter encrypted2_iter = iter(encrypted2);

        if (dest_size == 3)
        {

            size_t tile_size = min<size_t>(coeff_count, size_t(256));
            size_t num_tiles = coeff_count / tile_size;
#ifdef SEAL_DEBUG
            if (coeff_count % tile_size != 0)
            {
                throw invalid_argument("tile_size does not divide coeff_count");
            }
#endif

            ConstRNSIter encrypted2_0_iter(*encrypted2_iter[0], tile_size);
            ConstRNSIter encrypted2_1_iter(*encrypted2_iter[1], tile_size);
            RNSIter encrypted1_0_iter(*encrypted1_iter[0], tile_size);
            RNSIter encrypted1_1_iter(*encrypted1_iter[1], tile_size);
            RNSIter encrypted1_2_iter(*encrypted1_iter[2], tile_size);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, tile_size, pool);

            SEAL_ITERATE(coeff_modulus, coeff_modulus_size, [&](auto I) {
                SEAL_ITERATE(iter(size_t(0)), num_tiles, [&](SEAL_MAYBE_UNUSED auto J) {

                    dyadic_product_coeffmod(
                        encrypted1_1_iter[0], encrypted2_1_iter[0], tile_size, I, encrypted1_2_iter[0]);

                    dyadic_product_coeffmod(encrypted1_1_iter[0], encrypted2_0_iter[0], tile_size, I, temp);
                     
                    dyadic_product_coeffmod(
                        encrypted1_0_iter[0], encrypted2_1_iter[0], tile_size, I, encrypted1_1_iter[0]);
                     
                    add_poly_coeffmod(encrypted1_1_iter[0], temp, tile_size, I, encrypted1_1_iter[0]);

                    dyadic_product_coeffmod(
                        encrypted1_0_iter[0], encrypted2_0_iter[0], tile_size, I, encrypted1_0_iter[0]);

                    encrypted1_0_iter++;
                    encrypted1_1_iter++;
                    encrypted1_2_iter++;
                    encrypted2_0_iter++;
                    encrypted2_1_iter++;
                });
            });
        }
        else
        {
             
            SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp, dest_size, coeff_count, coeff_modulus_size, pool);

            SEAL_ITERATE(iter(size_t(0)), dest_size, [&](auto I) {

                size_t curr_encrypted1_last = min<size_t>(I, encrypted1_size - 1);
                size_t curr_encrypted2_first = min<size_t>(I, encrypted2_size - 1);
                size_t curr_encrypted1_first = I - curr_encrypted2_first;

                size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                auto shifted_encrypted1_iter = encrypted1_iter + curr_encrypted1_first;

                auto shifted_reversed_encrypted2_iter = reverse_iter(encrypted2_iter + curr_encrypted2_first);

                SEAL_ITERATE(iter(shifted_encrypted1_iter, shifted_reversed_encrypted2_iter), steps, [&](auto J) {

                    SEAL_ITERATE(iter(J, coeff_modulus, temp[I]), coeff_modulus_size, [&](auto K) {
                        SEAL_ALLOCATE_GET_COEFF_ITER(prod, coeff_count, pool);
                        dyadic_product_coeffmod(get<0, 0>(K), get<0, 1>(K), coeff_count, get<1>(K), prod);
                        add_poly_coeffmod(prod, get<2>(K), coeff_count, get<1>(K), get<2>(K));
                    });
                });
            });

            set_poly_array(temp, dest_size, coeff_count, coeff_modulus_size, encrypted1.data());
        }

        encrypted1.correction_factor() =
            multiply_uint_mod(encrypted1.correction_factor(), encrypted2.correction_factor(), parms.plain_modulus());
    }

    void Evaluator::square_inplace(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::bfv:
            bfv_square(encrypted, std::move(pool));
            break;

        case scheme_type::ckks:
            ckks_square(encrypted, std::move(pool));
            break;

        case scheme_type::bgv:
            bgv_square(encrypted, std::move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::bfv_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t base_q_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        uint64_t plain_modulus = parms.plain_modulus().value();

        auto rns_tool = context_data.rns_tool();
        size_t base_Bsk_size = rns_tool->base_Bsk()->size();
        size_t base_Bsk_m_tilde_size = rns_tool->base_Bsk_m_tilde()->size();

        if (encrypted_size != 2)
        {
            bfv_multiply(encrypted, encrypted, std::move(pool));
            return;
        }

        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        if (!product_fits_in(dest_size, coeff_count, base_Bsk_m_tilde_size))
        {
            throw logic_error("invalid parameters");
        }

        auto base_q = iter(parms.coeff_modulus());
        auto base_Bsk = iter(rns_tool->base_Bsk()->base());

        auto base_q_ntt_tables = iter(context_data.small_ntt_tables());
        auto base_Bsk_ntt_tables = iter(rns_tool->base_Bsk_ntt_tables());

        encrypted.resize(context_, context_data.parms_id(), dest_size);

        auto behz_extend_base_convert_to_ntt = [&](auto I) {

            set_poly(get<0>(I), coeff_count, base_q_size, get<1>(I));
            ntt_negacyclic_harvey_lazy(get<1>(I), base_q_size, base_q_ntt_tables);

            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count, base_Bsk_m_tilde_size, pool);

            rns_tool->fastbconv_m_tilde(get<0>(I), temp, pool);

            rns_tool->sm_mrq(temp, get<2>(I), pool);

            ntt_negacyclic_harvey_lazy(get<2>(I), base_Bsk_size, base_Bsk_ntt_tables);
        };

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted_q, encrypted_size, coeff_count, base_q_size, pool);

        SEAL_ALLOCATE_GET_POLY_ITER(encrypted_Bsk, encrypted_size, coeff_count, base_Bsk_size, pool);

        SEAL_ITERATE(iter(encrypted, encrypted_q, encrypted_Bsk), encrypted_size, behz_extend_base_convert_to_ntt);

        SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp_dest_q, dest_size, coeff_count, base_q_size, pool);
        SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp_dest_Bsk, dest_size, coeff_count, base_Bsk_size, pool);

        auto behz_ciphertext_square = [&](ConstPolyIter in_iter, ConstModulusIter base_iter, size_t base_size,
                                          PolyIter out_iter) {
             
            dyadic_product_coeffmod(in_iter[0], in_iter[0], base_size, base_iter, out_iter[0]);

            dyadic_product_coeffmod(in_iter[0], in_iter[1], base_size, base_iter, out_iter[1]);
            add_poly_coeffmod(out_iter[1], out_iter[1], base_size, base_iter, out_iter[1]);

            dyadic_product_coeffmod(in_iter[1], in_iter[1], base_size, base_iter, out_iter[2]);
        };

        behz_ciphertext_square(encrypted_q, base_q, base_q_size, temp_dest_q);
        behz_ciphertext_square(encrypted_Bsk, base_Bsk, base_Bsk_size, temp_dest_Bsk);

        inverse_ntt_negacyclic_harvey(temp_dest_q, dest_size, base_q_ntt_tables);
        inverse_ntt_negacyclic_harvey(temp_dest_Bsk, dest_size, base_Bsk_ntt_tables);

        SEAL_ITERATE(iter(temp_dest_q, temp_dest_Bsk, encrypted), dest_size, [&](auto I) {
             
            SEAL_ALLOCATE_GET_RNS_ITER(temp_q_Bsk, coeff_count, base_q_size + base_Bsk_size, pool);

            multiply_poly_scalar_coeffmod(get<0>(I), base_q_size, plain_modulus, base_q, temp_q_Bsk);

            multiply_poly_scalar_coeffmod(get<1>(I), base_Bsk_size, plain_modulus, base_Bsk, temp_q_Bsk + base_q_size);

            SEAL_ALLOCATE_GET_RNS_ITER(temp_Bsk, coeff_count, base_Bsk_size, pool);

            rns_tool->fast_floor(temp_q_Bsk, temp_Bsk, pool);

            rns_tool->fastbconv_sk(temp_Bsk, get<2>(I), pool);
        });
    }

    void Evaluator::ckks_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();

        if (encrypted_size != 2)
        {
            ckks_multiply(encrypted, encrypted, std::move(pool));
            return;
        }

        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        auto coeff_modulus = iter(parms.coeff_modulus());

        encrypted.resize(context_, context_data.parms_id(), dest_size);

        auto encrypted_iter = iter(encrypted);

        dyadic_product_coeffmod(
            encrypted_iter[1], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[2]);

        dyadic_product_coeffmod(
            encrypted_iter[0], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[1]);
        add_poly_coeffmod(encrypted_iter[1], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[1]);

        dyadic_product_coeffmod(
            encrypted_iter[0], encrypted_iter[0], coeff_modulus_size, coeff_modulus, encrypted_iter[0]);

        encrypted.scale() *= encrypted.scale();
        if (!is_scale_within_bounds(encrypted.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::bgv_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();

        if (encrypted_size != 2)
        {
            bgv_multiply(encrypted, encrypted, std::move(pool));
            return;
        }

        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        auto coeff_modulus = iter(parms.coeff_modulus());

        encrypted.resize(context_, context_data.parms_id(), dest_size);

        auto encrypted_iter = iter(encrypted);

        SEAL_ALLOCATE_ZERO_GET_POLY_ITER(temp, dest_size, coeff_count, coeff_modulus_size, pool);

        dyadic_product_coeffmod(
            encrypted_iter[1], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[2]);

        dyadic_product_coeffmod(
            encrypted_iter[0], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[1]);
        add_poly_coeffmod(encrypted_iter[1], encrypted_iter[1], coeff_modulus_size, coeff_modulus, encrypted_iter[1]);

        dyadic_product_coeffmod(
            encrypted_iter[0], encrypted_iter[0], coeff_modulus_size, coeff_modulus, encrypted_iter[0]);

        encrypted.correction_factor() =
            multiply_uint_mod(encrypted.correction_factor(), encrypted.correction_factor(), parms.plain_modulus());
    }

    void Evaluator::relinearize_internal(
        Ciphertext &encrypted, const RelinKeys &relin_keys, size_t destination_size, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (relin_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }

        size_t encrypted_size = encrypted.size();

        if (destination_size < 2 || destination_size > encrypted_size)
        {
            throw invalid_argument("destination_size must be at least 2 and less than or equal to current count");
        }
        if (relin_keys.size() < sub_safe(encrypted_size, size_t(2)))
        {
            throw invalid_argument("not enough relinearization keys");
        }

        if (destination_size == encrypted_size)
        {
            return;
        }

        size_t relins_needed = encrypted_size - destination_size;

        auto encrypted_iter = iter(encrypted);
        encrypted_iter += encrypted_size - 1;

        SEAL_ITERATE(iter(size_t(0)), relins_needed, [&](auto I) {
            this->switch_key_inplace(
                encrypted, *encrypted_iter, static_cast<const KSwitchKeys &>(relin_keys),
                RelinKeys::get_index(encrypted_size - 1 - I), pool);
        });

        encrypted.resize(context_, context_data_ptr->parms_id(), destination_size);
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_switch_scale_to_next(
        const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_data_ptr->parms().scheme() == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (context_data_ptr->parms().scheme() == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (context_data_ptr->parms().scheme() == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &next_context_data = *context_data.next_context_data();
        auto &next_parms = next_context_data.parms();
        auto rns_tool = context_data.rns_tool();

        size_t encrypted_size = encrypted.size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();

        Ciphertext encrypted_copy(pool);
        encrypted_copy = encrypted;

        switch (next_parms.scheme())
        {
        case scheme_type::bfv:
            SEAL_ITERATE(iter(encrypted_copy), encrypted_size, [&](auto I) {
                rns_tool->divide_and_round_q_last_inplace(I, pool);
            });
            break;

        case scheme_type::ckks:
            SEAL_ITERATE(iter(encrypted_copy), encrypted_size, [&](auto I) {
                rns_tool->divide_and_round_q_last_ntt_inplace(I, context_data.small_ntt_tables(), pool);
            });
            break;

        case scheme_type::bgv:
            SEAL_ITERATE(iter(encrypted_copy), encrypted_size, [&](auto I) {
                rns_tool->mod_t_and_divide_q_last_ntt_inplace(I, context_data.small_ntt_tables(), pool);
            });
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }

        destination.resize(context_, next_context_data.parms_id(), encrypted_size);
        SEAL_ITERATE(iter(encrypted_copy, destination), encrypted_size, [&](auto I) {
            set_poly(get<0>(I), coeff_count, next_coeff_modulus_size, get<1>(I));
        });

        destination.is_ntt_form() = encrypted.is_ntt_form();
        if (next_parms.scheme() == scheme_type::ckks)
        {
             
            destination.scale() =
                encrypted.scale() / static_cast<double>(context_data.parms().coeff_modulus().back().value());
        }
        else if (next_parms.scheme() == scheme_type::bgv)
        {
             
            destination.correction_factor() = multiply_uint_mod(
                encrypted.correction_factor(), rns_tool->inv_q_last_mod_t(), next_parms.plain_modulus());
        }
    }

    void Evaluator::mod_switch_drop_to_next(
        const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_data_ptr->parms().scheme() == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }

        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = next_context_data.parms();

        if (!is_scale_within_bounds(encrypted.scale(), next_context_data))
        {
            throw invalid_argument("scale out of bounds");
        }

        size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.size();

        if (!product_fits_in(encrypted_size, coeff_count, next_coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        auto drop_modulus_and_copy = [&](ConstPolyIter in_iter, PolyIter out_iter) {
            SEAL_ITERATE(iter(in_iter, out_iter), encrypted_size, [&](auto I) {
                SEAL_ITERATE(
                    iter(I), next_coeff_modulus_size, [&](auto J) { set_uint(get<0>(J), coeff_count, get<1>(J)); });
            });
        };

        if (&encrypted == &destination)
        {
             
            SEAL_ALLOCATE_GET_POLY_ITER(temp, encrypted_size, coeff_count, next_coeff_modulus_size, pool);

            drop_modulus_and_copy(encrypted, temp);

            destination.resize(context_, next_context_data.parms_id(), encrypted_size);

            set_poly_array(temp, encrypted_size, coeff_count, next_coeff_modulus_size, destination.data());
             
        }
        else
        {
             
            destination.resize(context_, next_context_data.parms_id(), encrypted_size);

            drop_modulus_and_copy(encrypted, destination);
        }
        destination.is_ntt_form() = true;
        destination.scale() = encrypted.scale();
        destination.correction_factor() = encrypted.correction_factor();
    }

    void Evaluator::mod_switch_drop_to_next(Plaintext &plain) const
    {
         
        auto context_data_ptr = context_.get_context_data(plain.parms_id());
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (!context_data_ptr->next_context_data())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }

        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = context_data_ptr->next_context_data()->parms();

        if (!is_scale_within_bounds(plain.scale(), next_context_data))
        {
            throw invalid_argument("scale out of bounds");
        }

        auto &next_coeff_modulus = next_parms.coeff_modulus();
        size_t next_coeff_modulus_size = next_coeff_modulus.size();
        size_t coeff_count = next_parms.poly_modulus_degree();

        auto dest_size = mul_safe(next_coeff_modulus_size, coeff_count);

        plain.parms_id() = parms_id_zero;
        plain.resize(dest_size);
        plain.parms_id() = next_context_data.parms_id();
    }

    void Evaluator::mod_switch_to_next(
        const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_.first_context_data()->parms().scheme())
        {
        case scheme_type::bfv:
             
            mod_switch_scale_to_next(encrypted, destination, std::move(pool));
            break;

        case scheme_type::ckks:
             
            mod_switch_drop_to_next(encrypted, destination, std::move(pool));
            break;

        case scheme_type::bgv:
            mod_switch_scale_to_next(encrypted, destination, std::move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (destination.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_switch_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(encrypted, pool);
        }
    }

    void Evaluator::mod_switch_to_inplace(Plaintext &plain, parms_id_type parms_id) const
    {
         
        auto context_data_ptr = context_.get_context_data(plain.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!context_.get_context_data(parms_id))
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(plain);
        }
    }

    void Evaluator::rescale_to_next(const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_.first_context_data()->parms().scheme())
        {
        case scheme_type::bfv:
             
        case scheme_type::bgv:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::ckks:
             
            mod_switch_scale_to_next(encrypted, destination, std::move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (destination.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::rescale_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::bfv:
             
        case scheme_type::bgv:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::ckks:
            while (encrypted.parms_id() != parms_id)
            {
                 
                mod_switch_scale_to_next(encrypted, encrypted, pool);
            }
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_reduce_to_next_inplace(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        mod_switch_drop_to_next(encrypted, encrypted, std::move(pool));
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::mod_reduce_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.parms_id() != parms_id)
        {
            mod_reduce_to_next_inplace(encrypted, pool);
        }
    }

    void Evaluator::multiply_many(
        const vector<Ciphertext> &encrypteds, const RelinKeys &relin_keys, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
         
        if (encrypteds.size() == 0)
        {
            throw invalid_argument("encrypteds vector must not be empty");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }

        auto context_data_ptr = context_.get_context_data(encrypteds[0].parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypteds is not valid for encryption parameters");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();

        if (parms.scheme() != scheme_type::bfv && parms.scheme() != scheme_type::bgv)
        {
            throw logic_error("unsupported scheme");
        }

        if (encrypteds.size() == 1)
        {
            destination = encrypteds[0];
            return;
        }

        vector<Ciphertext> product_vec;
        for (size_t i = 0; i < encrypteds.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            if (encrypteds[i].data() == encrypteds[i + 1].data())
            {
                square(encrypteds[i], temp);
            }
            else
            {
                multiply(encrypteds[i], encrypteds[i + 1], temp);
            }
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(std::move(temp));
        }
        if (encrypteds.size() & 1)
        {
            product_vec.emplace_back(encrypteds.back());
        }

        for (size_t i = 0; i < product_vec.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            multiply(product_vec[i], product_vec[i + 1], temp);
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(std::move(temp));
        }

        destination = product_vec.back();
    }

    void Evaluator::exponentiate_inplace(
        Ciphertext &encrypted, uint64_t exponent, const RelinKeys &relin_keys, MemoryPoolHandle pool) const
    {
         
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_.get_context_data(relin_keys.parms_id()))
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be 0");
        }

        if (exponent == 1)
        {
            return;
        }

        vector<Ciphertext> exp_vector(static_cast<size_t>(exponent), encrypted);
        multiply_many(exp_vector, relin_keys, encrypted, std::move(pool));
    }

    void Evaluator::add_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::bfv)
        {
            if (encrypted.is_ntt_form())
            {
                throw invalid_argument("BFV encrypted cannot be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BFV plain cannot be in NTT form");
            }
        }
        else if (parms.scheme() == scheme_type::ckks)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("CKKS encrypted must be in NTT form");
            }
            if (!plain.is_ntt_form())
            {
                throw invalid_argument("CKKS plain must be in NTT form");
            }
            if (encrypted.parms_id() != plain.parms_id())
            {
                throw invalid_argument("encrypted and plain parameter mismatch");
            }
            if (!are_same_scale(encrypted, plain))
            {
                throw invalid_argument("scale mismatch");
            }
        }
        else if (parms.scheme() == scheme_type::bgv)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("BGV encrypted must be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BGV plain cannot be in NTT form");
            }
        }

        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::bfv:
        {
            multiply_add_plain_with_scaling_variant(plain, context_data, *iter(encrypted));
            break;
        }

        case scheme_type::ckks:
        {
            RNSIter encrypted_iter(encrypted.data(), coeff_count);
            ConstRNSIter plain_iter(plain.data(), coeff_count);
            add_poly_coeffmod(encrypted_iter, plain_iter, coeff_modulus_size, coeff_modulus, encrypted_iter);
            break;
        }

        case scheme_type::bgv:
        {
            Plaintext plain_copy = plain;
            multiply_poly_scalar_coeffmod(
                plain.data(), plain.coeff_count(), encrypted.correction_factor(), parms.plain_modulus(),
                plain_copy.data());
            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), std::move(pool));
            RNSIter encrypted_iter(encrypted.data(), coeff_count);
            ConstRNSIter plain_iter(plain_copy.data(), coeff_count);
            add_poly_coeffmod(encrypted_iter, plain_iter, coeff_modulus_size, coeff_modulus, encrypted_iter);
            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::sub_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::bfv)
        {
            if (encrypted.is_ntt_form())
            {
                throw invalid_argument("BFV encrypted cannot be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BFV plain cannot be in NTT form");
            }
        }
        else if (parms.scheme() == scheme_type::ckks)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("CKKS encrypted must be in NTT form");
            }
            if (!plain.is_ntt_form())
            {
                throw invalid_argument("CKKS plain must be in NTT form");
            }
            if (encrypted.parms_id() != plain.parms_id())
            {
                throw invalid_argument("encrypted and plain parameter mismatch");
            }
            if (!are_same_scale(encrypted, plain))
            {
                throw invalid_argument("scale mismatch");
            }
        }
        else if (parms.scheme() == scheme_type::bgv)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("BGV encrypted must be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BGV plain cannot be in NTT form");
            }
        }

        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::bfv:
        {
            multiply_sub_plain_with_scaling_variant(plain, context_data, *iter(encrypted));
            break;
        }

        case scheme_type::ckks:
        {
            RNSIter encrypted_iter(encrypted.data(), coeff_count);
            ConstRNSIter plain_iter(plain.data(), coeff_count);
            sub_poly_coeffmod(encrypted_iter, plain_iter, coeff_modulus_size, coeff_modulus, encrypted_iter);
            break;
        }

        case scheme_type::bgv:
        {
            Plaintext plain_copy = plain;
            multiply_poly_scalar_coeffmod(
                plain.data(), plain.coeff_count(), encrypted.correction_factor(), parms.plain_modulus(),
                plain_copy.data());
            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), std::move(pool));
            RNSIter encrypted_iter(encrypted.data(), coeff_count);
            ConstRNSIter plain_iter(plain_copy.data(), coeff_count);
            sub_poly_coeffmod(encrypted_iter, plain_iter, coeff_modulus_size, coeff_modulus, encrypted_iter);
            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (encrypted.is_ntt_form() && plain.is_ntt_form())
        {
            multiply_plain_ntt(encrypted, plain);
        }
        else if (!encrypted.is_ntt_form() && !plain.is_ntt_form())
        {
            multiply_plain_normal(encrypted, plain, std::move(pool));
        }
        else if (encrypted.is_ntt_form() && !plain.is_ntt_form())
        {
            Plaintext plain_copy = plain;
            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), std::move(pool));
            multiply_plain_ntt(encrypted, plain_copy);
        }
        else
        {
            transform_to_ntt_inplace(encrypted);
            multiply_plain_ntt(encrypted, plain);
            transform_from_ntt_inplace(encrypted);
        }

#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::multiply_plain_normal(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
         
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();
        auto ntt_tables = iter(context_data.small_ntt_tables());

        size_t encrypted_size = encrypted.size();
        size_t plain_coeff_count = plain.coeff_count();
        size_t plain_nonzero_coeff_count = plain.nonzero_coeff_count();

        if (!product_fits_in(encrypted_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        if (plain_nonzero_coeff_count == 1)
        {
             
            size_t mono_exponent = plain.significant_coeff_count() - 1;

            if (plain[mono_exponent] >= plain_upper_half_threshold)
            {
                if (!context_data.qualifiers().using_fast_plain_lift)
                {
                     
                    SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_modulus_size, pool);

                    add_uint(plain_upper_half_increment, coeff_modulus_size, plain[mono_exponent], temp);
                    context_data.rns_tool()->base_q()->decompose(temp, pool);
                    negacyclic_multiply_poly_mono_coeffmod(
                        encrypted, encrypted_size, temp, mono_exponent, coeff_modulus, encrypted, pool);
                }
                else
                {

                    negacyclic_multiply_poly_mono_coeffmod(
                        encrypted, encrypted_size, plain[mono_exponent], mono_exponent, coeff_modulus, encrypted, pool);
                }
            }
            else
            {
                 
                negacyclic_multiply_poly_mono_coeffmod(
                    encrypted, encrypted_size, plain[mono_exponent], mono_exponent, coeff_modulus, encrypted, pool);
            }

            if (parms.scheme() == scheme_type::ckks)
            {
                encrypted.scale() *= plain.scale();
                if (!is_scale_within_bounds(encrypted.scale(), context_data))
                {
                    throw invalid_argument("scale out of bounds");
                }
            }

            return;
        }

        auto temp(allocate_zero_poly(coeff_count, coeff_modulus_size, pool));

        if (!context_data.qualifiers().using_fast_plain_lift)
        {
            StrideIter<uint64_t *> temp_iter(temp.get(), coeff_modulus_size);

            SEAL_ITERATE(iter(plain.data(), temp_iter), plain_coeff_count, [&](auto I) {
                auto plain_value = get<0>(I);
                if (plain_value >= plain_upper_half_threshold)
                {
                    add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
                }
                else
                {
                    *get<1>(I) = plain_value;
                }
            });

            context_data.rns_tool()->base_q()->decompose_array(temp_iter, coeff_count, pool);
        }
        else
        {

            RNSIter temp_iter(temp.get(), coeff_count);
            SEAL_ITERATE(iter(temp_iter, plain_upper_half_increment), coeff_modulus_size, [&](auto I) {
                SEAL_ITERATE(iter(get<0>(I), plain.data()), plain_coeff_count, [&](auto J) {
                    get<0>(J) =
                        SEAL_COND_SELECT(get<1>(J) >= plain_upper_half_threshold, get<1>(J) + get<1>(I), get<1>(J));
                });
            });
        }

        RNSIter temp_iter(temp.get(), coeff_count);
        ntt_negacyclic_harvey(temp_iter, coeff_modulus_size, ntt_tables);

        SEAL_ITERATE(iter(encrypted), encrypted_size, [&](auto I) {
            SEAL_ITERATE(iter(I, temp_iter, coeff_modulus, ntt_tables), coeff_modulus_size, [&](auto J) {
                 
                ntt_negacyclic_harvey_lazy(get<0>(J), get<3>(J));
                dyadic_product_coeffmod(get<0>(J), get<1>(J), coeff_count, get<2>(J), get<0>(J));
                inverse_ntt_negacyclic_harvey(get<0>(J), get<3>(J));
            });
        });

        if (parms.scheme() == scheme_type::ckks)
        {
            encrypted.scale() *= plain.scale();
            if (!is_scale_within_bounds(encrypted.scale(), context_data))
            {
                throw invalid_argument("scale out of bounds");
            }
        }
    }

    void Evaluator::multiply_plain_ntt(Ciphertext &encrypted_ntt, const Plaintext &plain_ntt) const
    {
         
        if (!plain_ntt.is_ntt_form())
        {
            throw invalid_argument("plain_ntt is not in NTT form");
        }
        if (encrypted_ntt.parms_id() != plain_ntt.parms_id())
        {
            throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
        }

        auto &context_data = *context_.get_context_data(encrypted_ntt.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_ntt_size = encrypted_ntt.size();

        if (!product_fits_in(encrypted_ntt_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        ConstRNSIter plain_ntt_iter(plain_ntt.data(), coeff_count);
        SEAL_ITERATE(iter(encrypted_ntt), encrypted_ntt_size, [&](auto I) {
            dyadic_product_coeffmod(I, plain_ntt_iter, coeff_modulus_size, coeff_modulus, I);
        });

        encrypted_ntt.scale() *= plain_ntt.scale();
        if (!is_scale_within_bounds(encrypted_ntt.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::transform_to_ntt_inplace(Plaintext &plain, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
         
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for the current context");
        }
        if (plain.is_ntt_form())
        {
            throw invalid_argument("plain is already in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t plain_coeff_count = plain.coeff_count();

        uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();

        auto ntt_tables = iter(context_data.small_ntt_tables());

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        plain.resize(coeff_count * coeff_modulus_size);
        RNSIter plain_iter(plain.data(), coeff_count);

        if (!context_data.qualifiers().using_fast_plain_lift)
        {

            SEAL_ALLOCATE_ZERO_GET_RNS_ITER(temp, coeff_modulus_size, coeff_count, pool);

            SEAL_ITERATE(iter(plain.data(), temp), plain_coeff_count, [&](auto I) {
                auto plain_value = get<0>(I);
                if (plain_value >= plain_upper_half_threshold)
                {
                    add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
                }
                else
                {
                    *get<1>(I) = plain_value;
                }
            });

            context_data.rns_tool()->base_q()->decompose_array(temp, coeff_count, pool);

            set_poly(temp, coeff_count, coeff_modulus_size, plain.data());
        }
        else
        {

            auto helper_iter = reverse_iter(plain_iter, plain_upper_half_increment);
            advance(helper_iter, -safe_cast<ptrdiff_t>(coeff_modulus_size - 1));

            SEAL_ITERATE(helper_iter, coeff_modulus_size, [&](auto I) {
                SEAL_ITERATE(iter(*plain_iter, get<0>(I)), plain_coeff_count, [&](auto J) {
                    get<1>(J) =
                        SEAL_COND_SELECT(get<0>(J) >= plain_upper_half_threshold, get<0>(J) + get<1>(I), get<0>(J));
                });
            });
        }

        ntt_negacyclic_harvey(plain_iter, coeff_modulus_size, ntt_tables);

        plain.parms_id() = parms_id;
    }

    void Evaluator::transform_to_ntt_inplace(Ciphertext &encrypted) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted is already in NTT form");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        auto ntt_tables = iter(context_data.small_ntt_tables());

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        ntt_negacyclic_harvey(encrypted, encrypted_size, ntt_tables);

        encrypted.is_ntt_form() = true;
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::transform_from_ntt_inplace(Ciphertext &encrypted_ntt) const
    {
         
        if (!is_metadata_valid_for(encrypted_ntt, context_) || !is_buffer_valid(encrypted_ntt))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted_ntt.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }
        if (!encrypted_ntt.is_ntt_form())
        {
            throw invalid_argument("encrypted_ntt is not in NTT form");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_ntt_size = encrypted_ntt.size();

        auto ntt_tables = iter(context_data.small_ntt_tables());

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        inverse_ntt_negacyclic_harvey(encrypted_ntt, encrypted_ntt_size, ntt_tables);

        encrypted_ntt.is_ntt_form() = false;
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted_ntt.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::apply_galois_inplace(
        Ciphertext &encrypted, uint32_t galois_elt, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const
    {
         
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        if (galois_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();
         
        auto galois_tool = context_.key_context_data()->galois_tool();

        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        if (!galois_keys.has_key(galois_elt))
        {
            throw invalid_argument("Galois key not present");
        }

        uint64_t m = mul_safe(static_cast<uint64_t>(coeff_count), uint64_t(2));

        if (!(galois_elt & 1) || unsigned_geq(galois_elt, m))
        {
            throw invalid_argument("Galois element is not valid");
        }
        if (encrypted_size > 2)
        {
            throw invalid_argument("encrypted size must be 2");
        }

        SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count, coeff_modulus_size, pool);

        if (parms.scheme() == scheme_type::bfv)
        {

            auto encrypted_iter = iter(encrypted);
            galois_tool->apply_galois(encrypted_iter[0], coeff_modulus_size, galois_elt, coeff_modulus, temp);

            set_poly(temp, coeff_count, coeff_modulus_size, encrypted.data(0));

            galois_tool->apply_galois(encrypted_iter[1], coeff_modulus_size, galois_elt, coeff_modulus, temp);
        }
        else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {

            auto encrypted_iter = iter(encrypted);
            galois_tool->apply_galois_ntt(encrypted_iter[0], coeff_modulus_size, galois_elt, temp);

            set_poly(temp, coeff_count, coeff_modulus_size, encrypted.data(0));

            galois_tool->apply_galois_ntt(encrypted_iter[1], coeff_modulus_size, galois_elt, temp);
        }
        else
        {
            throw logic_error("scheme not implemented");
        }

        set_zero_poly(coeff_count, coeff_modulus_size, encrypted.data(1));

        switch_key_inplace(
            encrypted, temp, static_cast<const KSwitchKeys &>(galois_keys), GaloisKeys::get_index(galois_elt), pool);
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
         
        if (encrypted.is_transparent())
        {
            throw logic_error("result ciphertext is transparent");
        }
#endif
    }

    void Evaluator::rotate_internal(
        Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const
    {
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_data_ptr->qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }
        if (galois_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        if (steps == 0)
        {
            return;
        }

        size_t coeff_count = context_data_ptr->parms().poly_modulus_degree();
        auto galois_tool = context_data_ptr->galois_tool();

        if (galois_keys.has_key(galois_tool->get_elt_from_step(steps)))
        {
             
            apply_galois_inplace(encrypted, galois_tool->get_elt_from_step(steps), galois_keys, std::move(pool));
        }
        else
        {
             
            vector<int> naf_steps = naf(steps);

            if (naf_steps.size() == 1)
            {
                throw invalid_argument("Galois key not present");
            }

            SEAL_ITERATE(naf_steps.cbegin(), naf_steps.size(), [&](auto step) {

                if (safe_cast<size_t>(abs(step)) != (coeff_count >> 1))
                {
                     
                    this->rotate_internal(encrypted, step, galois_keys, pool);
                }
            });
        }
    }

    void Evaluator::switch_key_inplace(
        Ciphertext &encrypted, ConstRNSIter target_iter, const KSwitchKeys &kswitch_keys, size_t kswitch_keys_index,
        MemoryPoolHandle pool) const
    {
        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto scheme = parms.scheme();

        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_iter)
        {
            throw invalid_argument("target_iter");
        }
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        if (kswitch_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }

        if (kswitch_keys_index >= kswitch_keys.data().size())
        {
            throw out_of_range("kswitch_keys_index");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (scheme == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }

        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        auto key_ntt_tables = iter(key_context_data.small_ntt_tables());
        auto modswitch_factors = key_context_data.rns_tool()->inv_q_last_mod_q();

        if (!product_fits_in(coeff_count, rns_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].data().size();

        for (auto &each_key : key_vector)
        {
            if (!is_metadata_valid_for(each_key, context_) || !is_buffer_valid(each_key))
            {
                throw invalid_argument("kswitch_keys is not valid for encryption parameters");
            }
        }

        SEAL_ALLOCATE_GET_RNS_ITER(t_target, coeff_count, decomp_modulus_size, pool);
        set_uint(target_iter, decomp_modulus_size * coeff_count, t_target);

        if (scheme == scheme_type::ckks || scheme == scheme_type::bgv)
        {
            inverse_ntt_negacyclic_harvey(t_target, decomp_modulus_size, key_ntt_tables);
        }

        auto t_poly_prod(allocate_zero_poly_array(key_component_count, coeff_count, rns_modulus_size, pool));

        SEAL_ITERATE(iter(size_t(0)), rns_modulus_size, [&](auto I) {
            size_t key_index = (I == decomp_modulus_size ? key_modulus_size - 1 : I);

            size_t lazy_reduction_summand_bound = size_t(SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX);
            size_t lazy_reduction_counter = lazy_reduction_summand_bound;

            auto t_poly_lazy(allocate_zero_poly_array(key_component_count, coeff_count, 2, pool));

            PolyIter accumulator_iter(t_poly_lazy.get(), 2, coeff_count);

            SEAL_ITERATE(iter(size_t(0)), decomp_modulus_size, [&](auto J) {
                SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);
                ConstCoeffIter t_operand;

                if ((scheme == scheme_type::ckks || scheme == scheme_type::bgv) && (I == J))
                {
                    t_operand = target_iter[J];
                }
                 
                else
                {
                     
                    if (key_modulus[J] <= key_modulus[key_index])
                    {
                        set_uint(t_target[J], coeff_count, t_ntt);
                    }
                     
                    else
                    {
                        modulo_poly_coeffs(t_target[J], coeff_count, key_modulus[key_index], t_ntt);
                    }
                     
                    ntt_negacyclic_harvey_lazy(t_ntt, key_ntt_tables[key_index]);
                    t_operand = t_ntt;
                }

                SEAL_ITERATE(iter(key_vector[J].data(), accumulator_iter), key_component_count, [&](auto K) {
                    if (!lazy_reduction_counter)
                    {
                        SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)), coeff_count, [&](auto L) {
                            unsigned long long qword[2]{ 0, 0 };
                            multiply_uint64(get<0>(L), get<1>(L), qword);

                            add_uint128(qword, get<2>(L).ptr(), qword);
                            get<2>(L)[0] = barrett_reduce_128(qword, key_modulus[key_index]);
                            get<2>(L)[1] = 0;
                        });
                    }
                    else
                    {
                         
                        SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)), coeff_count, [&](auto L) {
                            unsigned long long qword[2]{ 0, 0 };
                            multiply_uint64(get<0>(L), get<1>(L), qword);
                            add_uint128(qword, get<2>(L).ptr(), qword);
                            get<2>(L)[0] = qword[0];
                            get<2>(L)[1] = qword[1];
                        });
                    }
                });

                if (!--lazy_reduction_counter)
                {
                    lazy_reduction_counter = lazy_reduction_summand_bound;
                }
            });

            PolyIter t_poly_prod_iter(t_poly_prod.get() + (I * coeff_count), coeff_count, rns_modulus_size);

            SEAL_ITERATE(iter(accumulator_iter, t_poly_prod_iter), key_component_count, [&](auto K) {
                if (lazy_reduction_counter == lazy_reduction_summand_bound)
                {
                    SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
                        get<1>(L) = static_cast<uint64_t>(*get<0>(L));
                    });
                }
                else
                {
                     
                    SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
                        get<1>(L) = barrett_reduce_128(get<0>(L).ptr(), key_modulus[key_index]);
                    });
                }
            });
        });

        PolyIter t_poly_prod_iter(t_poly_prod.get(), coeff_count, rns_modulus_size);
        SEAL_ITERATE(iter(encrypted, t_poly_prod_iter), key_component_count, [&](auto I) {
            if (scheme == scheme_type::bgv)
            {
                const Modulus &plain_modulus = parms.plain_modulus();
                 
                uint64_t qk = key_modulus[key_modulus_size - 1].value();
                uint64_t qk_inv_qp = context_.key_context_data()->rns_tool()->inv_q_last_mod_t();

                CoeffIter t_last(get<1>(I)[decomp_modulus_size]);
                inverse_ntt_negacyclic_harvey(t_last, key_ntt_tables[key_modulus_size - 1]);

                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(k, coeff_count, pool);
                modulo_poly_coeffs(t_last, coeff_count, plain_modulus, k);
                negate_poly_coeffmod(k, coeff_count, plain_modulus, k);
                if (qk_inv_qp != 1)
                {
                    multiply_poly_scalar_coeffmod(k, coeff_count, qk_inv_qp, plain_modulus, k);
                }

                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(delta, coeff_count, pool);
                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(c_mod_qi, coeff_count, pool);
                SEAL_ITERATE(iter(I, key_modulus, modswitch_factors, key_ntt_tables), decomp_modulus_size, [&](auto J) {
                     
                    modulo_poly_coeffs(k, coeff_count, get<1>(J), delta);
                     
                    multiply_poly_scalar_coeffmod(delta, coeff_count, qk, get<1>(J), delta);

                    modulo_poly_coeffs(t_last, coeff_count, get<1>(J), c_mod_qi);

                    SEAL_ITERATE(iter(delta, c_mod_qi), coeff_count, [&](auto K) {
                        get<0>(K) = add_uint_mod(get<0>(K), get<1>(K), get<1>(J));
                    });
                    ntt_negacyclic_harvey(delta, get<3>(J));
                    SEAL_ITERATE(iter(delta, get<0, 1>(J)), coeff_count, [&](auto K) {
                        get<1>(K) = sub_uint_mod(get<1>(K), get<0>(K), get<1>(J));
                    });

                    multiply_poly_scalar_coeffmod(get<0, 1>(J), coeff_count, get<2>(J), get<1>(J), get<0, 1>(J));

                    add_poly_coeffmod(get<0, 1>(J), get<0, 0>(J), coeff_count, get<1>(J), get<0, 0>(J));
                });
            }
            else
            {
                 
                CoeffIter t_last(get<1>(I)[decomp_modulus_size]);
                inverse_ntt_negacyclic_harvey_lazy(t_last, key_ntt_tables[key_modulus_size - 1]);

                uint64_t qk = key_modulus[key_modulus_size - 1].value();
                uint64_t qk_half = qk >> 1;
                SEAL_ITERATE(t_last, coeff_count, [&](auto &J) {
                    J = barrett_reduce_64(J + qk_half, key_modulus[key_modulus_size - 1]);
                });

                SEAL_ITERATE(iter(I, key_modulus, key_ntt_tables, modswitch_factors), decomp_modulus_size, [&](auto J) {
                    SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);

                    uint64_t qi = get<1>(J).value();
                    if (qk > qi)
                    {
                         
                        modulo_poly_coeffs(t_last, coeff_count, get<1>(J), t_ntt);
                    }
                    else
                    {
                        set_uint(t_last, coeff_count, t_ntt);
                    }

                    uint64_t fix = qi - barrett_reduce_64(qk_half, get<1>(J));
                    SEAL_ITERATE(t_ntt, coeff_count, [fix](auto &K) { K += fix; });

                    uint64_t qi_lazy = qi << 1;  
                    if (scheme == scheme_type::ckks)
                    {
                         
                        ntt_negacyclic_harvey_lazy(t_ntt, get<2>(J));
#if SEAL_USER_MOD_BIT_COUNT_MAX > 60
                         
                        SEAL_ITERATE(
                            t_ntt, coeff_count, [&](auto &K) { K -= SEAL_COND_SELECT(K >= qi_lazy, qi_lazy, 0); });
#else
                         
                        qi_lazy = qi << 2;
#endif
                    }
                    else if (scheme == scheme_type::bfv)
                    {
                        inverse_ntt_negacyclic_harvey_lazy(get<0, 1>(J), get<2>(J));
                    }

                    SEAL_ITERATE(
                        iter(get<0, 1>(J), t_ntt), coeff_count, [&](auto K) { get<0>(K) += qi_lazy - get<1>(K); });

                    multiply_poly_scalar_coeffmod(get<0, 1>(J), coeff_count, get<3>(J), get<1>(J), get<0, 1>(J));
                    add_poly_coeffmod(get<0, 1>(J), get<0, 0>(J), coeff_count, get<1>(J), get<0, 0>(J));
                });
            }
        });
    }
}
