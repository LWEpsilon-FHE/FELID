// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/common.h"
#include "util/uintarith.h"
#include "util/uintarithmod.h"
#include "util/uintcore.h"

using namespace std;

namespace phantom::arith {
    bool try_invert_uint_mod(
            const uint64_t *operand, const uint64_t *modulus, size_t uint64_count, uint64_t *result) {
         
        int bit_count = get_significant_bit_count_uint(operand, uint64_count);
        if (bit_count == 0) {
            return false;
        }

        if (bit_count == 1) {
            set_uint(1, uint64_count, result);
            return true;
        }

        auto alloc_anchor = std::vector<uint64_t>(7 * uint64_count);

        uint64_t *numerator = alloc_anchor.data();
        set_uint(modulus, uint64_count, numerator);

        uint64_t *denominator = numerator + uint64_count;
        set_uint(operand, uint64_count, denominator);

        uint64_t *difference = denominator + uint64_count;

        int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
        int denominator_bits = get_significant_bit_count_uint(denominator, uint64_count);

        uint64_t *quotient = difference + uint64_count;

        uint64_t *invert_prior = quotient + uint64_count;
        set_zero_uint(uint64_count, invert_prior);
        bool invert_prior_positive = true;

        uint64_t *invert_curr = invert_prior + uint64_count;
        set_uint(1, uint64_count, invert_curr);
        bool invert_curr_positive = true;

        uint64_t *invert_next = invert_curr + uint64_count;
        bool invert_next_positive = true;

        while (true) {

            auto division_uint64_count = (size_t) (divide_round_up(numerator_bits, bits_per_uint64));

            int denominator_shift = numerator_bits - denominator_bits;
            left_shift_uint(denominator, denominator_shift, division_uint64_count, denominator);
            denominator_bits += denominator_shift;

            set_zero_uint(uint64_count, quotient);

            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits) {

                if (sub_uint(numerator, denominator, division_uint64_count, difference)) {

                    if (remaining_shifts == 0) {
                         
                        break;
                    }

                    add_uint(difference, numerator, division_uint64_count, difference);

                    left_shift_uint(quotient, 1, division_uint64_count, quotient);
                    remaining_shifts--;
                }

                *quotient |= 1;

                numerator_bits = get_significant_bit_count_uint(difference, division_uint64_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts) {

                    numerator_shift = remaining_shifts;
                }

                if (numerator_bits > 0) {
                    left_shift_uint(difference, numerator_shift, division_uint64_count, numerator);
                    numerator_bits += numerator_shift;
                } else {
                     
                    set_zero_uint(division_uint64_count, numerator);
                }

                left_shift_uint(quotient, numerator_shift, division_uint64_count, quotient);
                remaining_shifts -= numerator_shift;
            }

            right_shift_uint(denominator, denominator_shift, division_uint64_count, denominator);
            denominator_bits -= denominator_shift;

            if (numerator_bits == 0) {
                break;
            }

            right_shift_uint(numerator, denominator_shift, division_uint64_count, numerator);
            numerator_bits -= denominator_shift;

            multiply_truncate_uint(quotient, invert_curr, uint64_count, invert_next);
            invert_next_positive = !invert_curr_positive;
            if (invert_prior_positive == invert_next_positive) {

                add_uint(invert_prior, invert_next, uint64_count, invert_next);
            } else {

                uint64_t borrow = sub_uint(invert_prior, invert_next, uint64_count, invert_next);
                if (borrow == 0) {

                    invert_next_positive = invert_prior_positive;
                } else {

                    invert_next_positive = !invert_prior_positive;
                    negate_uint(invert_next, uint64_count, invert_next);
                }
            }

            swap(invert_prior, invert_curr);
            swap(invert_prior_positive, invert_curr_positive);
            swap(invert_curr, invert_next);
            swap(invert_curr_positive, invert_next_positive);

            swap(numerator, denominator);
            swap(numerator_bits, denominator_bits);
        }

        if (!is_equal_uint(denominator, uint64_count, 1)) {
             
            return false;
        }

        if (!invert_curr_positive && !is_zero_uint(invert_curr, uint64_count)) {
            sub_uint(modulus, invert_curr, uint64_count, invert_curr);
            invert_curr_positive = true;
        }

        set_uint(invert_curr, uint64_count, result);
        return true;
    }
}
