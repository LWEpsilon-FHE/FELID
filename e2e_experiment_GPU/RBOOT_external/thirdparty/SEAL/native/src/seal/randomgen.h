// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/dynarray.h"
#include "seal/memorymanager.h"
#include "seal/version.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace seal
{
    constexpr std::size_t prng_seed_uint64_count = 8;

    constexpr std::size_t prng_seed_byte_count = prng_seed_uint64_count * util::bytes_per_uint64;

    using prng_seed_type = std::array<std::uint64_t, prng_seed_uint64_count>;

    enum class prng_type : std::uint8_t
    {
        unknown = 0,

        blake2xb = 1,

        shake256 = 2
    };

    void random_bytes(seal_byte *buf, std::size_t count);

    SEAL_NODISCARD inline std::uint64_t random_uint64()
    {
        std::uint64_t result;
        random_bytes(reinterpret_cast<seal_byte *>(&result), sizeof(result));
        return result;
    }

    class UniformRandomGenerator;

    class UniformRandomGeneratorInfo
    {
        friend class UniformRandomGenerator;

    public:
         
        UniformRandomGeneratorInfo() = default;

        UniformRandomGeneratorInfo(prng_type type, prng_seed_type seed) : type_(type), seed_(std::move(seed))
        {}

        UniformRandomGeneratorInfo(const UniformRandomGeneratorInfo &copy) = default;

        UniformRandomGeneratorInfo &operator=(const UniformRandomGeneratorInfo &assign) = default;

        SEAL_NODISCARD inline bool operator==(const UniformRandomGeneratorInfo &compare) const noexcept
        {
            return (seed_ == compare.seed_) && (type_ == compare.type_);
        }

        SEAL_NODISCARD inline bool operator!=(const UniformRandomGeneratorInfo &compare) const noexcept
        {
            return !operator==(compare);
        }

        void clear() noexcept
        {
            type_ = prng_type::unknown;
            util::seal_memzero(seed_.data(), prng_seed_byte_count);
        }

        ~UniformRandomGeneratorInfo()
        {
            clear();
        }

        std::shared_ptr<UniformRandomGenerator> make_prng() const;

        SEAL_NODISCARD inline bool has_valid_prng_type() const noexcept
        {
            switch (type_)
            {
            case prng_type::blake2xb:

            case prng_type::shake256:

            case prng_type::unknown:
                return true;
            }
            return false;
        }

        SEAL_NODISCARD inline prng_type type() const noexcept
        {
            return type_;
        }

        SEAL_NODISCARD inline prng_type &type() noexcept
        {
            return type_;
        }

        SEAL_NODISCARD inline const prng_seed_type &seed() const noexcept
        {
            return seed_;
        }

        SEAL_NODISCARD inline prng_seed_type &seed() noexcept
        {
            return seed_;
        }

        SEAL_NODISCARD static inline std::streamoff SaveSize(
            compr_mode_type compr_mode = Serialization::compr_mode_default)
        {
            std::size_t members_size =
                Serialization::ComprSizeEstimate(sizeof(prng_type) + prng_seed_byte_count, compr_mode);
            return static_cast<std::streamoff>(sizeof(Serialization::SEALHeader) + members_size);
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            return UniformRandomGeneratorInfo::SaveSize(compr_mode);
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&UniformRandomGeneratorInfo::save_members, this, _1), save_size(compr_mode_type::none),
                stream, compr_mode, true);
        }

        inline std::streamoff load(std::istream &stream)
        {
            using namespace std::placeholders;
            UniformRandomGeneratorInfo new_info;
            auto in_size = Serialization::Load(
                std::bind(&UniformRandomGeneratorInfo::load_members, &new_info, _1, _2), stream, true);
            std::swap(*this, new_info);
            return in_size;
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&UniformRandomGeneratorInfo::save_members, this, _1), save_size(compr_mode_type::none), out,
                size, compr_mode, true);
        }

        inline std::streamoff load(const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            UniformRandomGeneratorInfo new_info;
            auto in_size = Serialization::Load(
                std::bind(&UniformRandomGeneratorInfo::load_members, &new_info, _1, _2), in, size, true);
            std::swap(*this, new_info);
            return in_size;
        }

    public:
        void save_members(std::ostream &stream) const;

        void load_members(std::istream &stream, SEALVersion version);

        prng_type type_ = prng_type::unknown;

        prng_seed_type seed_ = {};
    };

    class UniformRandomGenerator
    {
    public:
         
        UniformRandomGenerator(prng_seed_type seed)
            : seed_([&seed]() {
                   
                  DynArray<std::uint64_t> new_seed(
                      seed.size(), MemoryManager::GetPool(mm_prof_opt::mm_force_new, true));

                  std::copy(seed.cbegin(), seed.cend(), new_seed.begin());
                  return new_seed;
              }()),
              buffer_(buffer_size_, MemoryManager::GetPool(mm_prof_opt::mm_force_new, true)),
              buffer_begin_(buffer_.begin()), buffer_end_(buffer_.end()), buffer_head_(buffer_.end())
        {}

        SEAL_NODISCARD inline prng_seed_type seed() const noexcept
        {
            prng_seed_type ret{};
            std::copy(seed_.cbegin(), seed_.cend(), ret.begin());
            return ret;
        }

        void generate(std::size_t byte_count, seal_byte *destination);

        SEAL_NODISCARD inline std::uint32_t generate()
        {
            std::uint32_t result;
            generate(sizeof(result), reinterpret_cast<seal_byte *>(&result));
            return result;
        }

        inline void refresh()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            refill_buffer();
            buffer_head_ = buffer_begin_;
        }

        SEAL_NODISCARD inline UniformRandomGeneratorInfo info() const noexcept
        {
            UniformRandomGeneratorInfo result;
            std::copy_n(seed_.cbegin(), prng_seed_uint64_count, result.seed_.begin());
            result.type_ = type();
            return result;
        }

        virtual ~UniformRandomGenerator() = default;

    protected:
        SEAL_NODISCARD virtual prng_type type() const noexcept = 0;

        virtual void refill_buffer() = 0;

        const DynArray<std::uint64_t> seed_;

        const std::size_t buffer_size_ = 4096;

    private:
        DynArray<seal_byte> buffer_;

        std::mutex mutex_;

    protected:
        seal_byte *const buffer_begin_;

        seal_byte *const buffer_end_;

        seal_byte *buffer_head_;
    };

    class UniformRandomGeneratorFactory
    {
    public:
         
        UniformRandomGeneratorFactory() : use_random_seed_(true)
        {}

        UniformRandomGeneratorFactory(prng_seed_type default_seed)
            : default_seed_(default_seed), use_random_seed_(false)
        {}

        SEAL_NODISCARD auto create() -> std::shared_ptr<UniformRandomGenerator>
        {
            return use_random_seed_ ? create_impl([]() {
                prng_seed_type seed;
                random_bytes(reinterpret_cast<seal_byte *>(seed.data()), prng_seed_byte_count);
                return seed;
            }())
                                    : create_impl(default_seed_);
        }

        SEAL_NODISCARD auto create(prng_seed_type seed) -> std::shared_ptr<UniformRandomGenerator>
        {
            return create_impl(seed);
        }

        virtual ~UniformRandomGeneratorFactory() = default;

        static auto DefaultFactory() -> std::shared_ptr<UniformRandomGeneratorFactory>;

        SEAL_NODISCARD inline bool use_random_seed() noexcept
        {
            return use_random_seed_;
        }

        SEAL_NODISCARD inline prng_seed_type default_seed() noexcept
        {
            return default_seed_;
        }

    protected:
        SEAL_NODISCARD virtual auto create_impl(prng_seed_type seed) -> std::shared_ptr<UniformRandomGenerator> = 0;

    private:
        prng_seed_type default_seed_ = {};

        bool use_random_seed_ = false;
    };

    class Blake2xbPRNG : public UniformRandomGenerator
    {
    public:
         
        Blake2xbPRNG(prng_seed_type seed) : UniformRandomGenerator(seed)
        {}

        ~Blake2xbPRNG() = default;

    protected:
        SEAL_NODISCARD prng_type type() const noexcept override
        {
            return prng_type::blake2xb;
        }

        void refill_buffer() override;

    private:
        std::uint64_t counter_ = 0;
    };

    class Blake2xbPRNGFactory : public UniformRandomGeneratorFactory
    {
    public:
         
        Blake2xbPRNGFactory() : UniformRandomGeneratorFactory()
        {}

        Blake2xbPRNGFactory(prng_seed_type default_seed) : UniformRandomGeneratorFactory(default_seed)
        {}

        ~Blake2xbPRNGFactory() = default;

    protected:
        SEAL_NODISCARD auto create_impl(prng_seed_type seed) -> std::shared_ptr<UniformRandomGenerator> override
        {
            return std::make_shared<Blake2xbPRNG>(seed);
        }

    private:
    };

    class Shake256PRNG : public UniformRandomGenerator
    {
    public:
         
        Shake256PRNG(prng_seed_type seed) : UniformRandomGenerator(seed)
        {}

        ~Shake256PRNG() = default;

    protected:
        SEAL_NODISCARD prng_type type() const noexcept override
        {
            return prng_type::shake256;
        }

        void refill_buffer() override;

    private:
        std::uint64_t counter_ = 0;
    };

    class Shake256PRNGFactory : public UniformRandomGeneratorFactory
    {
    public:
         
        Shake256PRNGFactory() : UniformRandomGeneratorFactory()
        {}

        Shake256PRNGFactory(prng_seed_type default_seed) : UniformRandomGeneratorFactory(default_seed)
        {}

        ~Shake256PRNGFactory() = default;

    protected:
        SEAL_NODISCARD auto create_impl(prng_seed_type seed) -> std::shared_ptr<UniformRandomGenerator> override
        {
            return std::make_shared<Shake256PRNG>(seed);
        }

    private:
    };
}
