// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/pointer.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <stdexcept>

namespace seal
{
    namespace util
    {
         
        template <typename ValueType, typename RootType, typename ScalarType>
        class Arithmetic
        {
        public:
            ValueType add(const ValueType &a, const ValueType &b) const;

            ValueType sub(const ValueType &a, const ValueType &b) const;

            ValueType mul_root(const ValueType &a, const RootType &r) const;

            ValueType mul_scalar(const ValueType &a, const ScalarType &s) const;

            RootType mul_root_scalar(const RootType &r, const ScalarType &s) const;

            ValueType guard(const ValueType &a) const;
        };

        template <typename ValueType, typename RootType, typename ScalarType>
        class DWTHandler
        {
        public:
            DWTHandler()
            {}

            DWTHandler(const Arithmetic<ValueType, RootType, ScalarType> &num_struct) : arithmetic_(num_struct)
            {}

            void transform_to_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                 
                size_t n = size_t(1) << log_n;
                 
                RootType r;
                ValueType u;
                ValueType v;
                 
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                 
                std::size_t gap = n >> 1;
                std::size_t m = 1;

                for (; m < (n >> 1); m <<= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 4)
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j += 4)
                            {
                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);

                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);

                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);

                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap >>= 1;
                }

                if (scalar != nullptr)
                {
                    RootType scaled_r;
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                        u = arithmetic_.mul_scalar(arithmetic_.guard(values[0]), *scalar);
                        v = arithmetic_.mul_root(values[1], scaled_r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        u = arithmetic_.guard(values[0]);
                        v = arithmetic_.mul_root(values[1], r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
                }
            }

            void transform_from_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                 
                size_t n = size_t(1) << log_n;
                 
                RootType r;
                ValueType u;
                ValueType v;
                 
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                 
                std::size_t gap = 1;
                std::size_t m = n >> 1;

                for (; m > 1; m >>= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 4)
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j += 4)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap <<= 1;
                }

                if (scalar != nullptr)
                {
                    r = *++roots;
                    RootType scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                    x = values;
                    y = x + gap;
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);

                            u = arithmetic_.guard(*x);
                            v = *y;
                            *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                        }
                    }
                }
                else
                {
                    r = *++roots;
                    x = values;
                    y = x + gap;
                    if (gap < 4)
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);

                            u = *x;
                            v = *y;
                            *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                            *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                        }
                    }
                }
            }

        private:
            Arithmetic<ValueType, RootType, ScalarType> arithmetic_;
        };
    }  
}
