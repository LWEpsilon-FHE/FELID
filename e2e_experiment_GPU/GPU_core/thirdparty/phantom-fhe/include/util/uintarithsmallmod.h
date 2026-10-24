// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "modulus.h"
#include "defines.h"
#include "numth.h"
#include "uintarith.h"
#include <cstdint>
#include <type_traits>

namespace phantom::arith {
     
    [[nodiscard]] inline std::uint64_t increment_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        operand++;
        return operand - (modulus.value() &
                          static_cast<std::uint64_t>(-static_cast<std::int64_t>(operand >= modulus.value())));
    }

    [[nodiscard]] inline std::uint64_t decrement_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        std::int64_t carry = static_cast<std::int64_t>(operand == 0);
        return operand - 1 + (modulus.value() & static_cast<std::uint64_t>(-carry));
    }

    [[nodiscard]] inline std::uint64_t negate_uint_mod(std::uint64_t operand, const Modulus &modulus) {
        std::int64_t non_zero = static_cast<std::int64_t>(operand != 0);
        return (modulus.value() - operand) & static_cast<std::uint64_t>(-non_zero);
    }

    [[nodiscard]] inline std::uint64_t add_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
         
        operand1 += operand2;
        return operand1 >= modulus.value() ? operand1 - modulus.value() : operand1;
    }

    [[nodiscard]] inline std::uint64_t sub_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
        uint64_t temp;
        int64_t borrow = (int64_t) (sub_uint64_generic(operand1, operand2, 0, &temp));
        return (uint64_t) (temp) + (modulus.value() & (uint64_t) (-borrow));
    }

    template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>

    [[nodiscard]] inline std::uint64_t barrett_reduce_128(const T *input, const Modulus &modulus) {

        uint64_t tmp1, tmp3, carry;
        std::vector<uint64_t> tmp2(2);
        const std::uint64_t *const_ratio = modulus.const_ratio().data();

        multiply_uint64_hw64(input[0], const_ratio[0], &carry);

        multiply_uint64(input[0], const_ratio[1], tmp2.data());

        tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

        multiply_uint64(input[1], const_ratio[0], tmp2.data());

        carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

        tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

        tmp3 = input[0] - tmp1 * modulus.value();

        return tmp3 >= modulus.value() ? tmp3 - modulus.value() : tmp3;
    }

    template<typename T, typename = std::enable_if_t<is_uint64_v<T>>>

    [[nodiscard]] inline std::uint64_t barrett_reduce_64(T input, const Modulus &modulus) {

        uint64_t tmp[2];
        const std::uint64_t *const_ratio = modulus.const_ratio().data();
        multiply_uint64_hw64(input, const_ratio[1], tmp + 1);

        tmp[0] = input - tmp[1] * modulus.value();

        return tmp[0] >= modulus.value() ? tmp[0] - modulus.value() : tmp[0];
    }

    [[nodiscard]] inline std::uint64_t multiply_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, const Modulus &modulus) {
        uint64_t z[2];
        multiply_uint64(operand1, operand2, z);
        return barrett_reduce_128(z, modulus);
    }

    inline uint64_t compute_shoup(const uint64_t operand, const uint64_t modulus) {
         
        __uint128_t temp = operand;
        temp <<= 64;  
        return temp / modulus;
    }

    [[nodiscard]] inline std::uint64_t multiply_uint_mod_shoup(
            const uint64_t x, const uint64_t y, const uint64_t y_shoup, const Modulus &modulus) {
        uint64_t tmp1;
        const std::uint64_t p = modulus.value();
        multiply_uint64_hw64(x, y_shoup, &tmp1);
        const uint64_t tmp2 = y * x - tmp1 * p;
        return tmp2 >= p ? tmp2 - p : tmp2;
    }

    [[nodiscard]] inline std::uint64_t multiply_uint_mod_lazy(
            const uint64_t x, const uint64_t y, const uint64_t y_shoup, const Modulus &modulus) {
        uint64_t tmp1;
        const std::uint64_t p = modulus.value();
        multiply_uint64_hw64(x, y_shoup, &tmp1);
        return y * x - tmp1 * p;
    }

    inline void modulo_uint_inplace(uint64_t *value, const size_t value_uint64_count, const Modulus &modulus) {
        if (value_uint64_count == 1) {
            if (*value < modulus.value()) {
                return;
            } else {
                *value = barrett_reduce_64(*value, modulus);
            }
        }

        for (std::size_t i = value_uint64_count - 1; i--;) {
            value[i] = barrett_reduce_128(value + i, modulus);
            value[i + 1] = 0;
        }
    }

    [[nodiscard]] inline std::uint64_t modulo_uint(
            const std::uint64_t *value, const size_t value_uint64_count, const Modulus &modulus) {
        if (value_uint64_count == 1) {
             
            if (*value < modulus.value())
                return *value;
            else
                return barrett_reduce_64(*value, modulus);
        }

        uint64_t temp[2]{0, value[value_uint64_count - 1]};
        for (size_t k = value_uint64_count - 1; k--;) {
            temp[0] = value[k];
            temp[1] = barrett_reduce_128(temp, modulus);
        }

        return temp[1];
    }

    inline std::uint64_t multiply_add_uint_mod(
            std::uint64_t operand1, std::uint64_t operand2, std::uint64_t operand3, const Modulus &modulus) {
         
        uint64_t temp[2];
        multiply_uint64(operand1, operand2, temp);
        temp[1] += add_uint64(temp[0], operand3, temp);
        return barrett_reduce_128(temp, modulus);
    }

    inline std::uint64_t multiply_add_uint_mod(
            uint64_t operand1, uint64_t operand2, uint64_t operand2_shoup, std::uint64_t operand3,
            const Modulus &modulus) {
        return add_uint_mod(
                multiply_uint_mod_shoup(operand1, operand2, operand2_shoup, modulus),
                barrett_reduce_64(operand3, modulus), modulus);
    }

    inline bool try_invert_uint_mod(std::uint64_t operand, const Modulus &modulus, std::uint64_t &result) {
        return try_invert_uint_mod(operand, modulus.value(), result);
    }

    [[nodiscard]] std::uint64_t exponentiate_uint_mod(
            std::uint64_t operand, std::uint64_t exponent, const Modulus &modulus);

    void divide_uint_mod_inplace(
            std::uint64_t *numerator, const Modulus &modulus, std::size_t uint64_count, std::uint64_t *quotient);

    [[nodiscard]] std::uint64_t dot_product_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, std::size_t count,
            const Modulus &modulus);
}
