// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/common.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"
#include <algorithm>
#include <array>

using namespace std;

namespace seal
{
    namespace util
    {
        void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, const uint64_t *operand2,
            size_t operand2_uint64_count, size_t result_uint64_count, uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw invalid_argument("operand1");
            }
            if (!operand2 && operand2_uint64_count > 0)
            {
                throw invalid_argument("operand2");
            }
            if (!result_uint64_count)
            {
                throw invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (result != nullptr && (operand1 == result || operand2 == result))
            {
                throw invalid_argument("result cannot point to the same value as operand1 or operand2");
            }
#endif
             
            if (!operand1_uint64_count || !operand2_uint64_count)
            {
                 
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = *operand1 * *operand2;
                return;
            }

            operand1_uint64_count = get_significant_uint64_count_uint(operand1, operand1_uint64_count);
            operand2_uint64_count = get_significant_uint64_count_uint(operand2, operand2_uint64_count);

            if (operand1_uint64_count == 1)
            {
                multiply_uint(operand2, operand2_uint64_count, *operand1, result_uint64_count, result);
                return;
            }
            if (operand2_uint64_count == 1)
            {
                multiply_uint(operand1, operand1_uint64_count, *operand2, result_uint64_count, result);
                return;
            }

            set_zero_uint(result_uint64_count, result);

            size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
            for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
            {
                const uint64_t *inner_operand2 = operand2;
                uint64_t *inner_result = result++;
                uint64_t carry = 0;
                size_t operand2_index = 0;
                size_t operand2_index_max = min(operand2_uint64_count, result_uint64_count - operand1_index);
                for (; operand2_index < operand2_index_max; operand2_index++)
                {
                     
                    unsigned long long temp_result[2];
                    multiply_uint64(*operand1, *inner_operand2++, temp_result);
                    carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, temp_result);
                    unsigned long long temp;
                    carry += add_uint64(*inner_result, temp_result[0], 0, &temp);
                    *inner_result++ = temp;
                }

                if (operand1_index + operand2_index_max < result_uint64_count)
                {
                    *inner_result = carry;
                }

                operand1++;
            }
        }

        void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, uint64_t operand2, size_t result_uint64_count,
            uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw invalid_argument("operand1");
            }
            if (!result_uint64_count)
            {
                throw invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (result != nullptr && operand1 == result)
            {
                throw invalid_argument("result cannot point to the same value as operand1");
            }
#endif
             
            if (!operand1_uint64_count || !operand2)
            {
                 
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = *operand1 * operand2;
                return;
            }

            set_zero_uint(result_uint64_count, result);

            unsigned long long carry = 0;
            size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
            for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
            {
                unsigned long long temp_result[2];
                multiply_uint64(*operand1++, operand2, temp_result);
                unsigned long long temp;
                carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
                *result++ = temp;
            }

            if (operand1_index_max < result_uint64_count)
            {
                *result = carry;
            }
        }

        void divide_uint_inplace(
            uint64_t *numerator, const uint64_t *denominator, size_t uint64_count, uint64_t *quotient, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!numerator && uint64_count > 0)
            {
                throw invalid_argument("numerator");
            }
            if (!denominator && uint64_count > 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient && uint64_count > 0)
            {
                throw invalid_argument("quotient");
            }
            if (is_zero_uint(denominator, uint64_count) && uint64_count > 0)
            {
                throw invalid_argument("denominator");
            }
            if (quotient && (numerator == quotient || denominator == quotient))
            {
                throw invalid_argument("quotient cannot point to same value as numerator or denominator");
            }
#endif
            if (!uint64_count)
            {
                return;
            }

            set_zero_uint(uint64_count, quotient);

            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count_uint(denominator, uint64_count);

            if (numerator_bits < denominator_bits)
            {
                return;
            }

            uint64_count = safe_cast<size_t>(divide_round_up(numerator_bits, bits_per_uint64));

            if (uint64_count == 1)
            {
                *quotient = *numerator / *denominator;
                *numerator -= *quotient * *denominator;
                return;
            }

            auto alloc_anchor(allocate_uint(uint64_count << 1, pool));

            uint64_t *shifted_denominator = alloc_anchor.get();

            uint64_t *difference = shifted_denominator + uint64_count;

            int denominator_shift = numerator_bits - denominator_bits;
            left_shift_uint(denominator, denominator_shift, uint64_count, shifted_denominator);
            denominator_bits += denominator_shift;

            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {

                if (sub_uint(numerator, shifted_denominator, uint64_count, difference))
                {

                    if (remaining_shifts == 0)
                    {
                         
                        break;
                    }

                    add_uint(difference, numerator, uint64_count, difference);

                    left_shift_uint(quotient, 1, uint64_count, quotient);
                    remaining_shifts--;
                }

                quotient[0] |= 1;

                numerator_bits = get_significant_bit_count_uint(difference, uint64_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts)
                {

                    numerator_shift = remaining_shifts;
                }

                if (numerator_bits > 0)
                {
                    left_shift_uint(difference, numerator_shift, uint64_count, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                     
                    set_zero_uint(uint64_count, numerator);
                }

                left_shift_uint(quotient, numerator_shift, uint64_count, quotient);
                remaining_shifts -= numerator_shift;
            }

            if (numerator_bits > 0)
            {
                right_shift_uint(numerator, denominator_shift, uint64_count, numerator);
            }
        }

        void divide_uint128_uint64_inplace_generic(uint64_t *numerator, uint64_t denominator, uint64_t *quotient)
        {
#ifdef SEAL_DEBUG
            if (!numerator)
            {
                throw invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw invalid_argument("quotient cannot point to same value as numerator");
            }
#endif
             
            constexpr size_t uint64_count = 2;

            quotient[0] = 0;
            quotient[1] = 0;

            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);

            if (numerator_bits < denominator_bits)
            {
                return;
            }

            uint64_t shifted_denominator[uint64_count]{ denominator, 0 };

            uint64_t difference[uint64_count]{ 0, 0 };

            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint128(shifted_denominator, denominator_shift, shifted_denominator);
            denominator_bits += denominator_shift;

            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {

                if (sub_uint(numerator, shifted_denominator, uint64_count, difference))
                {

                    if (remaining_shifts == 0)
                    {
                         
                        break;
                    }

                    add_uint(difference, numerator, uint64_count, difference);

                    quotient[1] = (quotient[1] << 1) | (quotient[0] >> (bits_per_uint64 - 1));
                    quotient[0] <<= 1;
                    remaining_shifts--;
                }

                numerator_bits = get_significant_bit_count_uint(difference, uint64_count);

                int numerator_shift = min(denominator_bits - numerator_bits, remaining_shifts);

                numerator[0] = 0;
                numerator[1] = 0;

                if (numerator_bits > 0)
                {
                    left_shift_uint128(difference, numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }

                quotient[0] |= 1;

                left_shift_uint128(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            if (numerator_bits > 0)
            {
                right_shift_uint128(numerator, denominator_shift, numerator);
            }
        }

        void divide_uint192_inplace(uint64_t *numerator, uint64_t denominator, uint64_t *quotient)
        {
#ifdef SEAL_DEBUG
            if (!numerator)
            {
                throw invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw invalid_argument("quotient cannot point to same value as numerator");
            }
#endif
             
            size_t uint64_count = 3;

            quotient[0] = 0;
            quotient[1] = 0;
            quotient[2] = 0;

            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);

            if (numerator_bits < denominator_bits)
            {
                return;
            }

            uint64_count = safe_cast<size_t>(divide_round_up(numerator_bits, bits_per_uint64));

            if (uint64_count == 1)
            {
                *quotient = *numerator / denominator;
                *numerator -= *quotient * denominator;
                return;
            }

            vector<uint64_t> shifted_denominator(uint64_count, 0);
            shifted_denominator[0] = denominator;

            vector<uint64_t> difference(uint64_count);

            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint192(shifted_denominator.data(), denominator_shift, shifted_denominator.data());
            denominator_bits += denominator_shift;

            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {

                if (sub_uint(numerator, shifted_denominator.data(), uint64_count, difference.data()))
                {

                    if (remaining_shifts == 0)
                    {
                         
                        break;
                    }

                    add_uint(difference.data(), numerator, uint64_count, difference.data());

                    left_shift_uint192(quotient, 1, quotient);
                    remaining_shifts--;
                }

                quotient[0] |= 1;

                numerator_bits = get_significant_bit_count_uint(difference.data(), uint64_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts)
                {

                    numerator_shift = remaining_shifts;
                }

                if (numerator_bits > 0)
                {
                    left_shift_uint192(difference.data(), numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                     
                    set_zero_uint(uint64_count, numerator);
                }

                left_shift_uint192(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            if (numerator_bits > 0)
            {
                right_shift_uint192(numerator, denominator_shift, numerator);
            }
        }

        uint64_t exponentiate_uint_safe(uint64_t operand, uint64_t exponent)
        {
             
            if (exponent == 0)
            {
                return 1;
            }
            if (exponent == 1)
            {
                return operand;
            }

            uint64_t power = operand;
            uint64_t product = 0;
            uint64_t intermediate = 1;

            while (true)
            {
                if (exponent & 1)
                {
                    product = mul_safe(power, intermediate);
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = mul_safe(power, power);
                swap(product, power);
            }

            return intermediate;
        }

        uint64_t exponentiate_uint(uint64_t operand, uint64_t exponent)
        {
             
            if (exponent == 0)
            {
                return 1;
            }
            if (exponent == 1)
            {
                return operand;
            }

            uint64_t power = operand;
            uint64_t product = 0;
            uint64_t intermediate = 1;

            while (true)
            {
                if (exponent & 1)
                {
                    product = power * intermediate;
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = power * power;
                swap(product, power);
            }

            return intermediate;
        }
    }  
}
