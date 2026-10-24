// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define SEAL_ASSERT(condition)                                                                                         \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; \
        }                                                                                                              \
    }

#define _SEAL_STRINGIZE(x) #x
#define SEAL_STRINGIZE(x) _SEAL_STRINGIZE(x)

#define _SEAL_JOIN(M, N) M##N
#define SEAL_JOIN(M, N) _SEAL_JOIN(M, N)

static_assert(sizeof(double) == 8, "Require sizeof(double) == 8");

static_assert(sizeof(int) == 4, "Require sizeof(int) == 4");

static_assert(sizeof(unsigned long long) == 8, "Require sizeof(unsigned long long) == 8");

#define SEAL_MOD_BIT_COUNT_MAX 61
#define SEAL_MOD_BIT_COUNT_MIN 2

#define SEAL_INTERNAL_MOD_BIT_COUNT 61

#define SEAL_USER_MOD_BIT_COUNT_MAX 60
#define SEAL_USER_MOD_BIT_COUNT_MIN 2

#define SEAL_PLAIN_MOD_BIT_COUNT_MAX SEAL_USER_MOD_BIT_COUNT_MAX
#define SEAL_PLAIN_MOD_BIT_COUNT_MIN SEAL_USER_MOD_BIT_COUNT_MIN

#define SEAL_COEFF_MOD_COUNT_MAX 64
#define SEAL_COEFF_MOD_COUNT_MIN 1

#define SEAL_POLY_MOD_DEGREE_MAX 131072
#define SEAL_POLY_MOD_DEGREE_MIN 2

#define SEAL_CIPHERTEXT_SIZE_MAX 16
#define SEAL_CIPHERTEXT_SIZE_MIN 2

#if SEAL_MOD_BIT_COUNT_MAX > 32
#define SEAL_MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (SEAL_MOD_BIT_COUNT_MAX << 1)))
#define SEAL_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX (1 << (128 - (SEAL_INTERNAL_MOD_BIT_COUNT_MAX << 1)))
#define SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX (1 << (128 - (SEAL_USER_MOD_BIT_COUNT_MAX << 1)))
#else
#define SEAL_MULTIPLY_ACCUMULATE_MOD_MAX SIZE_MAX
#define SEAL_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX SIZE_MAX
#define SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX SIZE_MAX
#endif

#define SEAL_SYSTEM_OTHER 1
#define SEAL_SYSTEM_WINDOWS 2
#define SEAL_SYSTEM_UNIX_LIKE 3

#if defined(_WIN32)
#define SEAL_SYSTEM SEAL_SYSTEM_WINDOWS
#elif defined(__linux__) || defined(__FreeBSD__) || defined(EMSCRIPTEN) || (defined(__APPLE__) && defined(__MACH__))
#define SEAL_SYSTEM SEAL_SYSTEM_UNIX_LIKE
#else
#define SEAL_SYSTEM SEAL_SYSTEM_OTHER
#error "Unsupported system"
#endif

#define SEAL_COMPILER_MSVC 1
#define SEAL_COMPILER_CLANG 2
#define SEAL_COMPILER_GCC 3

#if defined(_MSC_VER)
#define SEAL_COMPILER SEAL_COMPILER_MSVC
#elif defined(__clang__)
#define SEAL_COMPILER SEAL_COMPILER_CLANG
#elif defined(__GNUC__) && !defined(__clang__)
#define SEAL_COMPILER SEAL_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

#include "seal/util/msvc.h"

#include "seal/util/clang.h"

#include "seal/util/gcc.h"

#ifdef SEAL_DEBUG
#define SEAL_DEBUG_V true
#else
#define SEAL_DEBUG_V false
#endif

#ifdef SEAL_USE_STD_BYTE
#include <cstddef>
namespace seal
{
    using seal_byte = std::byte;
}  
#else
namespace seal
{
    enum class seal_byte : unsigned char
    {
    };
}  
#endif

#ifndef SEAL_FORCE_INLINE
#define SEAL_FORCE_INLINE inline
#endif

#ifdef SEAL_USE_IF_CONSTEXPR
#define SEAL_IF_CONSTEXPR if constexpr
#else
#define SEAL_IF_CONSTEXPR if
#endif

#ifdef SEAL_USE_MAYBE_UNUSED
#define SEAL_MAYBE_UNUSED [[maybe_unused]]
#else
#define SEAL_MAYBE_UNUSED
#endif

#ifdef SEAL_USE_NODISCARD
#define SEAL_NODISCARD [[nodiscard]]
#else
#define SEAL_NODISCARD
#endif

#ifndef SEAL_USE_STD_FOR_EACH_N
#define SEAL_ITERATE seal::util::seal_for_each_n
#else
#define SEAL_ITERATE std::for_each_n
#endif

#ifndef SEAL_MALLOC
#define SEAL_MALLOC(size) (new seal_byte[size])
#endif

#ifndef SEAL_FREE
#define SEAL_FREE(ptr) (delete[] ptr)
#endif

#define SEAL_DEFAULT_PRNG_FACTORY SEAL_JOIN(SEAL_DEFAULT_PRNG, PRNGFactory)

#ifdef SEAL_USE_GAUSSIAN_NOISE
#define SEAL_NOISE_SAMPLER sample_poly_normal
#else
#define SEAL_NOISE_SAMPLER sample_poly_cbd
#endif

#ifndef SEAL_ADD_CARRY_UINT64
#define SEAL_ADD_CARRY_UINT64(operand1, operand2, carry, result) add_uint64_generic(operand1, operand2, carry, result)
#endif

#ifndef SEAL_SUB_BORROW_UINT64
#define SEAL_SUB_BORROW_UINT64(operand1, operand2, borrow, result) \
    sub_uint64_generic(operand1, operand2, borrow, result)
#endif

#ifndef SEAL_MULTIPLY_UINT64
#define SEAL_MULTIPLY_UINT64(operand1, operand2, result128) multiply_uint64_generic(operand1, operand2, result128)
#endif

#ifndef SEAL_DIVIDE_UINT128_UINT64
#define SEAL_DIVIDE_UINT128_UINT64(numerator, denominator, result) \
    divide_uint128_uint64_inplace_generic(numerator, denominator, result);
#endif

#ifndef SEAL_MULTIPLY_UINT64_HW64
#define SEAL_MULTIPLY_UINT64_HW64(operand1, operand2, hw64) multiply_uint64_hw64_generic(operand1, operand2, hw64)
#endif

#ifndef SEAL_MSB_INDEX_UINT64
#define SEAL_MSB_INDEX_UINT64(result, value) get_msb_index_generic(result, value)
#endif

#define SEAL_ASSERT_TYPE(obj, expected, message)                                                                    \
    do                                                                                                              \
    {                                                                                                               \
        static_assert(                                                                                              \
            std::is_same<decltype(obj), expected>::value,                                                           \
            "In " __FILE__ ":" SEAL_STRINGIZE(__LINE__) " expected " SEAL_STRINGIZE(expected) " (message: " message \
                                                                                              ")");                 \
    } while (false)

#define SEAL_ALLOCATE_GET_PTR_ITER(name, type, size, pool)                               \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(seal::util::allocate<type>(size, pool)); \
    seal::util::PtrIter<type *> name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get());

#define SEAL_ALLOCATE_GET_STRIDE_ITER(name, type, size, stride, pool)                                                  \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(seal::util::allocate<type>(seal::util::mul_safe(size, stride), pool)); \
    seal::util::StrideIter<type *> name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get(), stride);

#define SEAL_ALLOCATE_GET_POLY_ITER(name, poly_count, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(                                                     \
        seal::util::allocate_poly_array(poly_count, poly_modulus_degree, coeff_modulus_size, pool)); \
    seal::util::PolyIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get(), poly_modulus_degree, coeff_modulus_size);

#define SEAL_ALLOCATE_ZERO_GET_POLY_ITER(name, poly_count, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(                                                          \
        seal::util::allocate_zero_poly_array(poly_count, poly_modulus_degree, coeff_modulus_size, pool)); \
    seal::util::PolyIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get(), poly_modulus_degree, coeff_modulus_size);

#define SEAL_ALLOCATE_GET_RNS_ITER(name, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(                                        \
        seal::util::allocate_poly(poly_modulus_degree, coeff_modulus_size, pool));      \
    seal::util::RNSIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get(), poly_modulus_degree);

#define SEAL_ALLOCATE_ZERO_GET_RNS_ITER(name, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(                                             \
        seal::util::allocate_zero_poly(poly_modulus_degree, coeff_modulus_size, pool));      \
    seal::util::RNSIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get(), poly_modulus_degree);

#define SEAL_ALLOCATE_GET_COEFF_ITER(name, poly_modulus_degree, pool)                                  \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(seal::util::allocate_uint(poly_modulus_degree, pool)); \
    seal::util::CoeffIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get());

#define SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(name, poly_modulus_degree, pool)                                  \
    auto SEAL_JOIN(_seal_temp_alloc_, __LINE__)(seal::util::allocate_zero_uint(poly_modulus_degree, pool)); \
    seal::util::CoeffIter name(SEAL_JOIN(_seal_temp_alloc_, __LINE__).get());

#define SEAL_COND_SELECT(cond, if_true, if_false) (cond ? if_true : if_false)
