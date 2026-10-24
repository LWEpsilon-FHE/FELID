// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/serialization.h"
#include "seal/version.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include "seal/util/pointer.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <type_traits>
#ifdef SEAL_USE_MSGSL
#include "gsl/span"
#endif

namespace seal
{
    class Ciphertext;

    namespace util
    {
        namespace ztools
        {
            template <typename SizeT>
            SEAL_NODISCARD SizeT deflate_size_bound(SizeT in_size);
        }  
    }  

    template <typename T>
    class DynArray
    {
        friend class Ciphertext;

    public:
        using type = T;

        DynArray(MemoryPoolHandle pool = MemoryManager::GetPool()) : pool_(std::move(pool))
        {
            if (!pool_)
            {
                throw std::invalid_argument("pool is uninitialized");
            }
        }

        explicit DynArray(std::size_t size, MemoryPoolHandle pool = MemoryManager::GetPool()) : pool_(std::move(pool))
        {
            if (!pool_)
            {
                throw std::invalid_argument("pool is uninitialized");
            }

            resize(size);
        }

        explicit DynArray(std::size_t capacity, std::size_t size, MemoryPoolHandle pool = MemoryManager::GetPool())
            : pool_(std::move(pool))
        {
            if (!pool_)
            {
                throw std::invalid_argument("pool is uninitialized");
            }
            if (capacity < size)
            {
                throw std::invalid_argument("capacity cannot be smaller than size");
            }

            reserve(capacity);
            resize(size);
        }

        explicit DynArray(
            util::Pointer<T> &&ptr, std::size_t capacity, std::size_t size, bool fill_zero,
            MemoryPoolHandle pool = MemoryManager::GetPool())
            : pool_(std::move(pool)), capacity_(capacity)
        {
            if (!ptr && capacity)
            {
                throw std::invalid_argument("ptr cannot be null");
            }
            if (!pool_)
            {
                throw std::invalid_argument("pool is uninitialized");
            }
            if (capacity < size)
            {
                throw std::invalid_argument("capacity cannot be smaller than size");
            }

            data_ = std::move(ptr);

            resize(size, fill_zero);
        }

        explicit DynArray(
            util::Pointer<T> &&ptr, std::size_t size, bool fill_zero, MemoryPoolHandle pool = MemoryManager::GetPool())
            : DynArray(std::move(ptr), size, size, fill_zero, std::move(pool))
        {}
#ifdef SEAL_USE_MSGSL
         
        explicit DynArray(
            gsl::span<const T> values, std::size_t capacity, MemoryPoolHandle pool = MemoryManager::GetPool())
            : DynArray(capacity, values.size(), std::move(pool))
        {
            std::copy(values.begin(), values.end(), data_.get());
        }

        explicit DynArray(gsl::span<const T> values, MemoryPoolHandle pool = MemoryManager::GetPool())
            : DynArray(values, values.size(), std::move(pool))
        {}
#endif
         
        DynArray(const DynArray<T> &copy)
            : pool_(MemoryManager::GetPool()), capacity_(copy.size_), size_(copy.size_),
              data_(util::allocate<T>(copy.size_, pool_))
        {
             
            std::copy(copy.cbegin(), copy.cend(), begin());
        }

        DynArray(DynArray<T> &&source) noexcept
            : pool_(std::move(source.pool_)), capacity_(source.capacity_), size_(source.size_),
              data_(std::move(source.data_))
        {}

        ~DynArray()
        {
            release();
        }

        SEAL_NODISCARD inline T *begin() noexcept
        {
            return data_.get();
        }

        SEAL_NODISCARD inline const T *begin() const noexcept
        {
            return cbegin();
        }

        SEAL_NODISCARD inline const T *cbegin() const noexcept
        {
            return data_.get();
        }

        SEAL_NODISCARD inline T *end() noexcept
        {
            return begin() + size_;
        }

        SEAL_NODISCARD inline const T *end() const noexcept
        {
            return cend();
        }

        SEAL_NODISCARD inline const T *cend() const noexcept
        {
            return cbegin() + size_;
        }
#ifdef SEAL_USE_MSGSL
         
        SEAL_NODISCARD inline gsl::span<T> span()
        {
            return gsl::span<T>(begin(), size_);
        }

        SEAL_NODISCARD inline gsl::span<const T> span() const
        {
            return gsl::span<const T>(cbegin(), size_);
        }
#endif
         
        SEAL_NODISCARD inline const T &at(std::size_t index) const
        {
            if (index >= size_)
            {
                throw std::out_of_range("index must be within [0, size)");
            }
            return data_[index];
        }

        SEAL_NODISCARD inline T &at(std::size_t index)
        {
            if (index >= size_)
            {
                throw std::out_of_range("index must be within [0, size)");
            }
            return data_[index];
        }

        SEAL_NODISCARD inline const T &operator[](std::size_t index) const
        {
            return data_[index];
        }

        SEAL_NODISCARD inline T &operator[](std::size_t index)
        {
            return data_[index];
        }

        SEAL_NODISCARD inline bool empty() const noexcept
        {
            return (size_ == 0);
        }

        SEAL_NODISCARD inline std::size_t max_size() const noexcept
        {
            return (std::numeric_limits<std::size_t>::max)();
        }

        SEAL_NODISCARD inline std::size_t size() const noexcept
        {
            return size_;
        }

        SEAL_NODISCARD inline std::size_t capacity() const noexcept
        {
            return capacity_;
        }

        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return pool_;
        }

        inline void release() noexcept
        {
            capacity_ = 0;
            size_ = 0;
            data_.release();
        }

        inline void clear() noexcept
        {
            size_ = 0;
        }

        inline void reserve(std::size_t capacity)
        {
            std::size_t copy_size = std::min<>(capacity, size_);

            auto new_data(util::allocate<T>(capacity, pool_));
            std::copy_n(cbegin(), copy_size, new_data.get());
            std::swap(data_, new_data);

            capacity_ = capacity;
            size_ = copy_size;
        }

        inline void shrink_to_fit()
        {
            reserve(size_);
        }

        inline void resize(std::size_t size, bool fill_zero = true)
        {
            if (size <= capacity_)
            {

                if (size > size_ && fill_zero)
                {
                    std::fill(end(), begin() + size, T(0));
                }

                size_ = size;

                return;
            }

            auto new_data(util::allocate<T>(size, pool_));
            std::copy(cbegin(), cend(), new_data.get());
            if (fill_zero)
            {
                std::fill(new_data.get() + size_, new_data.get() + size, T(0));
            }
            std::swap(data_, new_data);

            capacity_ = size;
            size_ = size;
        }
#ifdef SEAL_USE_MSGSL
         
        inline DynArray<T> &operator=(gsl::span<const T> values)
        {
             
            resize(0, false);
            resize(values.size(), false);

            std::copy(values.begin(), values.end(), begin());

            return *this;
        }
#endif
         
        inline DynArray<T> &operator=(const DynArray<T> &assign)
        {
             
            if (this == &assign)
            {
                return *this;
            }

            resize(assign.size_);

            std::copy(assign.cbegin(), assign.cend(), begin());

            return *this;
        }

        DynArray<T> &operator=(DynArray<T> &&assign) noexcept
        {
            capacity_ = assign.capacity_;
            size_ = assign.size_;
            data_ = std::move(assign.data_);
            pool_ = std::move(assign.pool_);

            return *this;
        }

        SEAL_NODISCARD inline std::streamoff save_size(
            compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            std::size_t members_size = Serialization::ComprSizeEstimate(
                util::add_safe(
                    sizeof(std::uint64_t),  
                    util::mul_safe(size_, sizeof(T))),  
                compr_mode);

            return util::safe_cast<std::streamoff>(util::add_safe(sizeof(Serialization::SEALHeader), members_size));
        }

        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&DynArray<T>::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        inline std::streamoff load(std::istream &stream, std::size_t in_size_bound = 0)
        {
            using namespace std::placeholders;
            return Serialization::Load(
                std::bind(&DynArray<T>::load_members, this, _1, _2, in_size_bound), stream, false);
        }

        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&DynArray<T>::save_members, this, _1), save_size(compr_mode_type::none), out, size,
                compr_mode, false);
        }

        inline std::streamoff load(const seal_byte *in, std::size_t size, std::size_t in_size_bound = 0)
        {
            using namespace std::placeholders;
            return Serialization::Load(
                std::bind(&DynArray<T>::load_members, this, _1, _2, in_size_bound), in, size, false);
        }

    private:
        void save_members(std::ostream &stream) const
        {
            auto old_except_mask = stream.exceptions();
            try
            {
                 
                stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

                std::uint64_t size64 = size_;
                stream.write(reinterpret_cast<const char *>(&size64), sizeof(std::uint64_t));
                if (size_)
                {
                    stream.write(
                        reinterpret_cast<const char *>(cbegin()),
                        util::safe_cast<std::streamsize>(util::mul_safe(size_, sizeof(T))));
                }
            }
            catch (const std::ios_base::failure &)
            {
                stream.exceptions(old_except_mask);
                throw std::runtime_error("I/O error");
            }
            catch (...)
            {
                stream.exceptions(old_except_mask);
                throw;
            }
            stream.exceptions(old_except_mask);
        }

        void load_members(std::istream &stream, SEAL_MAYBE_UNUSED SEALVersion version, std::size_t in_size_bound)
        {
            auto old_except_mask = stream.exceptions();
            try
            {
                 
                stream.exceptions(std::ios_base::badbit | std::ios_base::failbit);

                std::uint64_t size64 = 0;
                stream.read(reinterpret_cast<char *>(&size64), sizeof(std::uint64_t));

                if (in_size_bound && util::unsigned_gt(size64, in_size_bound))
                {
                    throw std::logic_error("unexpected size");
                }

                resize(util::safe_cast<std::size_t>(size64));

                if (size_)
                {
                    stream.read(
                        reinterpret_cast<char *>(begin()),
                        util::safe_cast<std::streamsize>(util::mul_safe(size_, sizeof(T))));
                }
            }
            catch (const std::ios_base::failure &)
            {
                stream.exceptions(old_except_mask);
                throw std::runtime_error("I/O error");
            }
            catch (...)
            {
                stream.exceptions(old_except_mask);
                throw;
            }
            stream.exceptions(old_except_mask);
        }

        MemoryPoolHandle pool_;

        std::size_t capacity_ = 0;

        std::size_t size_ = 0;

        util::Pointer<T> data_;
    };
}
