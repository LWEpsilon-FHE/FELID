// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/serialization.h"
#include "seal/util/defines.h"
#include "seal/util/streambuf.h"
#include <ios>
#include <iostream>

namespace seal
{
     
    template <class T>
    class Serializable
    {
        friend class KeyGenerator;
        friend class Encryptor;

    public:
         
        Serializable(const Serializable<T> &copy) = default;

        Serializable(Serializable<T> &&source) = default;

        Serializable &operator=(Serializable<T> &&assign) = default;

        Serializable &operator=(const Serializable<T> &copy) = default;

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return obj_.save_size(compr_mode);
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return obj_.save(stream, compr_mode);
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return obj_.save(out, size, compr_mode);
        }

    private:
        Serializable(T &&obj) : obj_(std::move(obj))
        {}

        T obj_;
    };
}
