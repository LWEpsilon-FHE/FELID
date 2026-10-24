// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/dynarray.h"
#include "seal/memorymanager.h"
#include "seal/serialization.h"
#include "seal/util/common.h"
#include "seal/util/streambuf.h"
#include "seal/util/ztools.h"
#include <stdexcept>
#include <typeinfo>

using namespace std;
using namespace seal::util;

namespace seal
{

    constexpr compr_mode_type Serialization::compr_mode_default;

    constexpr uint16_t Serialization::seal_magic;

    constexpr uint8_t Serialization::seal_header_size;

    namespace
    {
        [[noreturn]] void expressive_rethrow_on_ios_base_failure(const ostream &stream)
        {
            if (!stream.rdbuf())
            {
                throw runtime_error("I/O error: output stream has no associated buffer");
            }

            auto &rdbuf_ref = *stream.rdbuf();
            if (typeid(rdbuf_ref).hash_code() == typeid(ArrayPutBuffer).hash_code())
            {
                auto buffer = reinterpret_cast<ArrayPutBuffer *>(stream.rdbuf());

                if (buffer->at_end())
                {
                     
                    throw runtime_error("I/O error: insufficient output buffer");
                }
            }

            throw runtime_error("I/O error");
        }

        [[noreturn]] void expressive_rethrow_on_ios_base_failure(const istream &stream)
        {
            if (!stream.rdbuf())
            {
                throw runtime_error("I/O error: input stream has no associated buffer");
            }

            if (stream.eof())
            {
                auto &rdbuf_ref = *stream.rdbuf();
                if (typeid(rdbuf_ref).hash_code() == typeid(ArrayGetBuffer).hash_code())
                {
                     
                    throw runtime_error("I/O error: input buffer ended unexpectedly");
                }
                else
                {
                     
                    throw runtime_error("I/O error: input stream ended unexpectedly");
                }
            }

            throw runtime_error("I/O error");
        }
    }  

    size_t Serialization::ComprSizeEstimate(size_t in_size, compr_mode_type compr_mode)
    {
        if (!IsSupportedComprMode(compr_mode))
        {
            throw invalid_argument("unsupported compression mode");
        }

        switch (compr_mode)
        {
#ifdef SEAL_USE_ZSTD
        case compr_mode_type::zstd:
            return ztools::zstd_deflate_size_bound(in_size);
#endif
#ifdef SEAL_USE_ZLIB
        case compr_mode_type::zlib:
            return ztools::zlib_deflate_size_bound(in_size);
#endif
        case compr_mode_type::none:
             
            return in_size;

        default:
            throw invalid_argument("unsupported compression mode");
        }
    }

    streamoff Serialization::SaveHeader(const SEALHeader &header, ostream &stream)
    {
        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            stream.write(reinterpret_cast<const char *>(&header), sizeof(SEALHeader));
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            expressive_rethrow_on_ios_base_failure(stream);
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        return static_cast<streamoff>(sizeof(SEALHeader));
    }

    streamoff Serialization::LoadHeader(istream &stream, SEALHeader &header, bool try_upgrade_if_invalid)
    {
        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            stream.read(reinterpret_cast<char *>(&header), sizeof(SEALHeader));

            if (try_upgrade_if_invalid && !IsValidHeader(header))
            {
                 
                legacy_headers::SEALHeader_3_4 header_3_4(header);

                SEALHeader new_header;
                new_header.version_major = 3;
                new_header.version_minor = 4;

                new_header.compr_mode = header_3_4.compr_mode;
                new_header.size = header_3_4.size;

                if (IsValidHeader(new_header))
                {
                    header = new_header;
                }
            }
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            expressive_rethrow_on_ios_base_failure(stream);
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        return static_cast<streamoff>(sizeof(SEALHeader));
    }

    streamoff Serialization::SaveHeader(const SEALHeader &header, seal_byte *out, size_t size)
    {
        if (!out)
        {
            throw invalid_argument("out cannot be null");
        }
        if (size < sizeof(SEALHeader))
        {
            throw invalid_argument("insufficient size");
        }
        if (!fits_in<streamsize>(size))
        {
            throw invalid_argument("size is too large");
        }
        ArrayPutBuffer apbuf(reinterpret_cast<char *>(out), static_cast<streamsize>(size));
        ostream stream(&apbuf);
        return SaveHeader(header, stream);
    }

    streamoff Serialization::LoadHeader(
        const seal_byte *in, size_t size, SEALHeader &header, bool try_upgrade_if_invalid)
    {
        if (!in)
        {
            throw invalid_argument("in cannot be null");
        }
        if (size < sizeof(SEALHeader))
        {
            throw invalid_argument("insufficient size");
        }
        if (!fits_in<streamsize>(size))
        {
            throw invalid_argument("size is too large");
        }
        ArrayGetBuffer agbuf(reinterpret_cast<const char *>(in), static_cast<streamsize>(size));
        istream stream(&agbuf);
        return LoadHeader(stream, header, try_upgrade_if_invalid);
    }

    streamoff Serialization::Save(
        function<void(ostream &)> save_members, streamoff raw_size, ostream &stream, compr_mode_type compr_mode,
        bool clear_buffers)
    {
        if (!save_members)
        {
            throw invalid_argument("save_members is invalid");
        }
        if (raw_size < static_cast<streamoff>(sizeof(SEALHeader)))
        {
            throw invalid_argument("raw_size is too small");
        }
        if (!IsSupportedComprMode(compr_mode))
        {
            throw invalid_argument("unsupported compression mode");
        }

        streamoff out_size = 0;

        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            auto stream_start_pos = stream.tellp();

            SEALHeader header;

            switch (compr_mode)
            {
            case compr_mode_type::none:
                 
                header.compr_mode = compr_mode;
                header.size = safe_cast<uint64_t>(raw_size);
                SaveHeader(header, stream);

                save_members(stream);
                break;
#ifdef SEAL_USE_ZLIB
            case compr_mode_type::zlib:
            {

                SafeByteBuffer safe_buffer(
                    ztools::zlib_deflate_size_bound(raw_size - static_cast<streamoff>(sizeof(SEALHeader))),
                    clear_buffers);
                iostream temp_stream(&safe_buffer);
                temp_stream.exceptions(ios_base::badbit | ios_base::failbit);
                save_members(temp_stream);

                auto safe_pool(MemoryManager::GetPool(mm_prof_opt::mm_force_new, clear_buffers));

                DynArray<seal_byte> safe_buffer_array(
                    Pointer<seal_byte>::Aliasing(safe_buffer.data()), safe_buffer.size(),
                    static_cast<size_t>(temp_stream.tellp()), false, safe_pool);

                ztools::zlib_write_header_deflate_buffer(
                    safe_buffer_array, reinterpret_cast<void *>(&header), stream, safe_pool);
                break;
            }
#endif
#ifdef SEAL_USE_ZSTD
            case compr_mode_type::zstd:
            {

                SafeByteBuffer safe_buffer(
                    ztools::zstd_deflate_size_bound(raw_size - static_cast<streamoff>(sizeof(SEALHeader))),
                    clear_buffers);
                iostream temp_stream(&safe_buffer);
                temp_stream.exceptions(ios_base::badbit | ios_base::failbit);
                save_members(temp_stream);

                auto safe_pool(MemoryManager::GetPool(mm_prof_opt::mm_force_new, clear_buffers));

                DynArray<seal_byte> safe_buffer_array(
                    Pointer<seal_byte>::Aliasing(safe_buffer.data()), safe_buffer.size(),
                    static_cast<size_t>(temp_stream.tellp()), false, safe_pool);

                ztools::zstd_write_header_deflate_buffer(
                    safe_buffer_array, reinterpret_cast<void *>(&header), stream, safe_pool);
                break;
            }
#endif
            default:
                throw invalid_argument("unsupported compression mode");
            }

            auto stream_end_pos = stream.tellp();
            out_size = stream_end_pos - stream_start_pos;
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            expressive_rethrow_on_ios_base_failure(stream);
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        return out_size;
    }

    streamoff Serialization::Load(
        function<void(istream &, SEALVersion)> load_members, istream &stream, bool clear_buffers)
    {
        if (!load_members)
        {
            throw invalid_argument("load_members is invalid");
        }

        streamoff in_size = 0;
        SEALHeader header;

        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            auto stream_start_pos = stream.tellg();

            LoadHeader(stream, header);
            if (!IsCompatibleVersion(header))
            {
                throw logic_error("incompatible version");
            }
            if (!IsValidHeader(header))
            {
                throw logic_error("loaded SEALHeader is invalid");
            }

            SEALVersion version{ header.version_major, header.version_minor, 0, 0 };

            switch (header.compr_mode)
            {
            case compr_mode_type::none:
                 
                load_members(stream, version);
                if (header.size != safe_cast<uint64_t>(stream.tellg() - stream_start_pos))
                {
                    throw logic_error("invalid data size");
                }
                break;
#ifdef SEAL_USE_ZLIB
            case compr_mode_type::zlib:
            {
                auto compr_size = header.size - safe_cast<uint64_t>(stream.tellg() - stream_start_pos);

                SafeByteBuffer safe_buffer(safe_cast<streamsize>(compr_size), clear_buffers);

                iostream temp_stream(&safe_buffer);
                temp_stream.exceptions(ios_base::badbit | ios_base::failbit);

                auto safe_pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, clear_buffers);

                if (ztools::zlib_inflate_stream(stream, safe_cast<streamoff>(compr_size), temp_stream, safe_pool))
                {
                    throw logic_error("stream decompression failed");
                }
                load_members(temp_stream, version);
                break;
            }
#endif
#ifdef SEAL_USE_ZSTD
            case compr_mode_type::zstd:
            {
                auto compr_size = header.size - safe_cast<uint64_t>(stream.tellg() - stream_start_pos);

                SafeByteBuffer safe_buffer(safe_cast<streamsize>(compr_size), clear_buffers);

                iostream temp_stream(&safe_buffer);
                temp_stream.exceptions(ios_base::badbit | ios_base::failbit);

                auto safe_pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, clear_buffers);

                if (ztools::zstd_inflate_stream(stream, safe_cast<streamoff>(compr_size), temp_stream, safe_pool))
                {
                    throw logic_error("stream decompression failed");
                }
                load_members(temp_stream, version);
                break;
            }
#endif
            default:
                throw invalid_argument("unsupported compression mode");
            }

            in_size = safe_cast<streamoff>(header.size);
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            expressive_rethrow_on_ios_base_failure(stream);
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        return in_size;
    }

    streamoff Serialization::Save(
        function<void(ostream &)> save_members, streamoff raw_size, seal_byte *out, size_t size,
        compr_mode_type compr_mode, bool clear_buffers)
    {
        if (!out)
        {
            throw invalid_argument("out cannot be null");
        }
        if (size < sizeof(SEALHeader))
        {
            throw invalid_argument("insufficient size");
        }
        if (!fits_in<streamsize>(size))
        {
            throw invalid_argument("size is too large");
        }
        ArrayPutBuffer apbuf(reinterpret_cast<char *>(out), static_cast<streamsize>(size));
        ostream stream(&apbuf);
        return Save(save_members, raw_size, stream, compr_mode, clear_buffers);
    }

    streamoff Serialization::Load(
        function<void(istream &, SEALVersion)> load_members, const seal_byte *in, size_t size, bool clear_buffers)
    {
        if (!in)
        {
            throw invalid_argument("in cannot be null");
        }
        if (size < sizeof(SEALHeader))
        {
            throw invalid_argument("insufficient size");
        }
        if (!fits_in<streamsize>(size))
        {
            throw invalid_argument("size is too large");
        }
        ArrayGetBuffer agbuf(reinterpret_cast<const char *>(in), static_cast<streamsize>(size));
        istream stream(&agbuf);
        return Load(load_members, stream, clear_buffers);
    }
}
