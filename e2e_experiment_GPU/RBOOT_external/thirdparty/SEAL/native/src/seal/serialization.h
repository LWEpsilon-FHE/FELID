// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/version.h"
#include "seal/util/defines.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>

namespace seal
{
     
    enum class compr_mode_type : std::uint8_t
    {
         
        none = 0,
#ifdef SEAL_USE_ZLIB
         
        zlib = 1,
#endif
#ifdef SEAL_USE_ZSTD
         
        zstd = 2,
#endif
    };

    class Serialization
    {
    public:
         
#if defined(SEAL_USE_ZSTD)
        static constexpr compr_mode_type compr_mode_default = compr_mode_type::zstd;
#elif defined(SEAL_USE_ZLIB)
        static constexpr compr_mode_type compr_mode_default = compr_mode_type::zlib;
#else
        static constexpr compr_mode_type compr_mode_default = compr_mode_type::none;
#endif
         
        static constexpr std::uint16_t seal_magic = 0xA15E;

        static constexpr std::uint8_t seal_header_size = 0x10;

        struct SEALHeader
        {
            std::uint16_t magic = seal_magic;

            std::uint8_t header_size = seal_header_size;

            std::uint8_t version_major = static_cast<std::uint8_t>(SEAL_VERSION_MAJOR);

            std::uint8_t version_minor = static_cast<std::uint8_t>(SEAL_VERSION_MINOR);

            compr_mode_type compr_mode = compr_mode_type::none;

            std::uint16_t reserved = 0;

            std::uint64_t size = 0;
        };

        static_assert(sizeof(SEALHeader) == seal_header_size, "");

        SEAL_NODISCARD static bool IsSupportedComprMode(std::uint8_t compr_mode) noexcept
        {
            switch (compr_mode)
            {
            case static_cast<std::uint8_t>(compr_mode_type::none):
                 
#ifdef SEAL_USE_ZLIB
            case static_cast<std::uint8_t>(compr_mode_type::zlib):
                 
#endif
#ifdef SEAL_USE_ZSTD
            case static_cast<std::uint8_t>(compr_mode_type::zstd):
#endif
                return true;
            }
            return false;
        }

        SEAL_NODISCARD static inline bool IsSupportedComprMode(compr_mode_type compr_mode) noexcept
        {
            return IsSupportedComprMode(static_cast<uint8_t>(compr_mode));
        }

        SEAL_NODISCARD static std::size_t ComprSizeEstimate(std::size_t in_size, compr_mode_type compr_mode);

        SEAL_NODISCARD static bool IsCompatibleVersion(const SEALHeader &header) noexcept
        {
             
            if (header.version_major == SEAL_VERSION_MAJOR && header.version_minor == SEAL_VERSION_MINOR)
            {
                return true;
            }

            if (header.version_major != SEAL_VERSION_MAJOR)
            {
                return false;
            }

            if (header.version_major == 3 && (header.version_minor == 4 || header.version_minor == 5))
            {
                return true;
            }

            return false;
        }

        SEAL_NODISCARD static bool IsValidHeader(const SEALHeader &header) noexcept
        {
            if (header.magic != seal_magic)
            {
                return false;
            }
            if (header.header_size != seal_header_size)
            {
                return false;
            }
            if (!IsCompatibleVersion(header))
            {
                return false;
            }
            if (!IsSupportedComprMode(static_cast<uint8_t>(header.compr_mode)))
            {
                return false;
            }
            return true;
        }

        static std::streamoff SaveHeader(const SEALHeader &header, std::ostream &stream);

        static std::streamoff LoadHeader(std::istream &stream, SEALHeader &header, bool try_upgrade_if_invalid = true);

        static std::streamoff SaveHeader(const SEALHeader &header, seal_byte *out, std::size_t size);

        static std::streamoff LoadHeader(
            const seal_byte *in, std::size_t size, SEALHeader &header, bool try_upgrade_if_invalid = true);

        static std::streamoff Save(
            std::function<void(std::ostream &)> save_members, std::streamoff raw_size, std::ostream &stream,
            compr_mode_type compr_mode, bool clear_buffers);

        static std::streamoff Load(
            std::function<void(std::istream &, SEALVersion)> load_members, std::istream &stream, bool clear_buffers);

        static std::streamoff Save(
            std::function<void(std::ostream &)> save_members, std::streamoff raw_size, seal_byte *out, std::size_t size,
            compr_mode_type compr_mode, bool clear_buffers);

        static std::streamoff Load(
            std::function<void(std::istream &, SEALVersion)> load_members, const seal_byte *in, std::size_t size,
            bool clear_buffers);

    private:
        Serialization() = delete;
    };

    namespace legacy_headers
    {
         
        struct SEALHeader_3_4
        {
            std::uint16_t magic = Serialization::seal_magic;

            std::uint8_t zero_byte = 0x00;

            compr_mode_type compr_mode = compr_mode_type::none;

            std::uint32_t size = 0;

            std::uint64_t reserved = 0;

            SEALHeader_3_4 &operator=(const Serialization::SEALHeader assign)
            {
                std::memcpy(this, &assign, Serialization::seal_header_size);
                return *this;
            }

            SEALHeader_3_4() = default;

            SEALHeader_3_4(const Serialization::SEALHeader &copy)
            {
                operator=(copy);
            }
        };
    }  
}
