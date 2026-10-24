// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/util/defines.h"
#include "seal/util/globals.h"
#include "seal/util/mempool.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifndef _M_CEE
#include <mutex>
#include <thread>
#endif

namespace seal
{
     
    class MemoryPoolHandle
    {
    public:
         
        MemoryPoolHandle() = default;

        MemoryPoolHandle(std::shared_ptr<util::MemoryPool> pool) noexcept : pool_(std::move(pool))
        {}

        MemoryPoolHandle(const MemoryPoolHandle &copy) noexcept
        {
            operator=(copy);
        }

        MemoryPoolHandle(MemoryPoolHandle &&source) noexcept
        {
            operator=(std::move(source));
        }

        inline MemoryPoolHandle &operator=(const MemoryPoolHandle &assign) noexcept
        {
            pool_ = assign.pool_;
            return *this;
        }

        inline MemoryPoolHandle &operator=(MemoryPoolHandle &&assign) noexcept
        {
            pool_ = std::move(assign.pool_);
            return *this;
        }

        SEAL_NODISCARD inline static MemoryPoolHandle Global() noexcept
        {
            return util::global_variables::global_memory_pool;
        }
#ifndef _M_CEE
         
        SEAL_NODISCARD inline static MemoryPoolHandle ThreadLocal() noexcept
        {
            return util::global_variables::tls_memory_pool;
        }
#endif
         
        SEAL_NODISCARD inline static MemoryPoolHandle New(bool clear_on_destruction = false)
        {
            return MemoryPoolHandle(std::make_shared<util::MemoryPoolMT>(clear_on_destruction));
        }

        SEAL_NODISCARD inline operator util::MemoryPool &() const
        {
            if (!pool_)
            {
                throw std::logic_error("pool not initialized");
            }
            return *pool_.get();
        }

        SEAL_NODISCARD inline std::size_t pool_count() const noexcept
        {
            return !pool_ ? std::size_t(0) : pool_->pool_count();
        }

        SEAL_NODISCARD inline std::size_t alloc_byte_count() const noexcept
        {
            return !pool_ ? std::size_t(0) : pool_->alloc_byte_count();
        }

        SEAL_NODISCARD inline long use_count() const noexcept
        {
            return !pool_ ? 0 : pool_.use_count();
        }

        SEAL_NODISCARD inline explicit operator bool() const noexcept
        {
            return pool_.operator bool();
        }

        inline bool operator==(const MemoryPoolHandle &compare) noexcept
        {
            return pool_ == compare.pool_;
        }

        inline bool operator!=(const MemoryPoolHandle &compare) noexcept
        {
            return pool_ != compare.pool_;
        }

    private:
        std::shared_ptr<util::MemoryPool> pool_ = nullptr;
    };

    using mm_prof_opt_t = std::uint64_t;

    enum mm_prof_opt : mm_prof_opt_t
    {
        mm_default = 0x0,
        mm_force_global = 0x1,
        mm_force_new = 0x2,
        mm_force_thread_local = 0x4
    };

    class MMProf
    {
    public:
         
        MMProf() = default;

        virtual ~MMProf() noexcept
        {}

        virtual MemoryPoolHandle get_pool(mm_prof_opt_t) = 0;

    private:
    };

    class MMProfGlobal : public MMProf
    {
    public:
         
        MMProfGlobal() = default;

        virtual ~MMProfGlobal() noexcept override
        {}

        SEAL_NODISCARD inline virtual MemoryPoolHandle get_pool(mm_prof_opt_t) override
        {
            return MemoryPoolHandle::Global();
        }

    private:
    };

    class MMProfNew : public MMProf
    {
    public:
         
        MMProfNew() = default;

        virtual ~MMProfNew() noexcept override
        {}

        SEAL_NODISCARD inline virtual MemoryPoolHandle get_pool(mm_prof_opt_t) override
        {
            return MemoryPoolHandle::New();
        }

    private:
    };

    class MMProfFixed : public MMProf
    {
    public:
         
        MMProfFixed(MemoryPoolHandle pool) : pool_(std::move(pool))
        {
            if (!pool_)
            {
                throw std::invalid_argument("pool is uninitialized");
            }
        }

        virtual ~MMProfFixed() noexcept override
        {}

        SEAL_NODISCARD inline virtual MemoryPoolHandle get_pool(mm_prof_opt_t) override
        {
            return pool_;
        }

    private:
        MemoryPoolHandle pool_;
    };
#ifndef _M_CEE
     
    class MMProfThreadLocal : public MMProf
    {
    public:
         
        MMProfThreadLocal() = default;

        virtual ~MMProfThreadLocal() noexcept override
        {}

        SEAL_NODISCARD inline virtual MemoryPoolHandle get_pool(mm_prof_opt_t) override
        {
            return MemoryPoolHandle::ThreadLocal();
        }

    private:
    };
#endif
     
    class MemoryManager
    {
        friend class MMProfGuard;

    public:
        MemoryManager() = delete;

        static inline std::unique_ptr<MMProf> SwitchProfile(MMProf *&&mm_prof) noexcept
        {
#ifndef _M_CEE
            std::lock_guard<std::mutex> switching_lock(switch_mutex_);
#endif
            return SwitchProfileThreadUnsafe(std::move(mm_prof));
        }

        static inline std::unique_ptr<MMProf> SwitchProfile(std::unique_ptr<MMProf> &&mm_prof) noexcept
        {
#ifndef _M_CEE
            std::lock_guard<std::mutex> switch_lock(switch_mutex_);
#endif
            return SwitchProfileThreadUnsafe(std::move(mm_prof));
        }

        template <typename... Args>
        SEAL_NODISCARD static inline MemoryPoolHandle GetPool(mm_prof_opt_t prof_opt, Args &&... args)
        {
            switch (prof_opt)
            {
            case mm_prof_opt::mm_force_global:
                return MemoryPoolHandle::Global();

            case mm_prof_opt::mm_force_new:
                return MemoryPoolHandle::New(std::forward<Args>(args)...);
#ifndef _M_CEE
            case mm_prof_opt::mm_force_thread_local:
                return MemoryPoolHandle::ThreadLocal();
#endif
            default:
#ifdef SEAL_DEBUG
            {
                auto pool = GetMMProf()->get_pool(prof_opt);
                if (!pool)
                {
                    throw std::logic_error("cannot return uninitialized pool");
                }
                return pool;
            }
#endif
                return GetMMProf()->get_pool(prof_opt);
            }
        }

        SEAL_NODISCARD static inline MemoryPoolHandle GetPool()
        {
            return GetPool(mm_prof_opt::mm_default);
        }

    private:
        SEAL_NODISCARD static inline std::unique_ptr<MMProf> SwitchProfileThreadUnsafe(MMProf *&&mm_prof)
        {
            if (!mm_prof)
            {
                throw std::invalid_argument("mm_prof cannot be null");
            }
            auto ret_mm_prof = std::move(GetMMProf());
            GetMMProf().reset(mm_prof);
            return ret_mm_prof;
        }

        SEAL_NODISCARD static inline std::unique_ptr<MMProf> SwitchProfileThreadUnsafe(
            std::unique_ptr<MMProf> &&mm_prof)
        {
            if (!mm_prof)
            {
                throw std::invalid_argument("mm_prof cannot be null");
            }
            std::swap(GetMMProf(), mm_prof);
            return std::move(mm_prof);
        }

        SEAL_NODISCARD static inline std::unique_ptr<MMProf> &GetMMProf()
        {
            static std::unique_ptr<MMProf> mm_prof{ new MMProfGlobal };
            return mm_prof;
        }
#ifndef _M_CEE
        static std::mutex switch_mutex_;
#endif
    };
#ifndef _M_CEE
     
    class MMProfGuard
    {
    public:
         
        MMProfGuard(std::unique_ptr<MMProf> &&mm_prof, bool start_locked = true) noexcept
            : mm_switch_lock_(MemoryManager::switch_mutex_, std::defer_lock)
        {
            if (start_locked)
            {
                lock(std::move(mm_prof));
            }
            else
            {
                old_prof_ = std::move(mm_prof);
            }
        }

        MMProfGuard(MMProf *&&mm_prof, bool start_locked = true) noexcept
            : mm_switch_lock_(MemoryManager::switch_mutex_, std::defer_lock)
        {
            if (start_locked)
            {
                lock(std::move(mm_prof));
            }
            else
            {
                old_prof_.reset(std::move(mm_prof));
            }
        }

        inline bool try_lock()
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            if (!mm_switch_lock_.try_lock())
            {
                return false;
            }
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(old_prof_));
            return true;
        }

#ifdef _MSC_VER
        _Acquires_lock_(mm_switch_lock_)
#endif
            inline void lock()
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            mm_switch_lock_.lock();
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(old_prof_));
        }

        inline bool try_lock(std::unique_ptr<MMProf> &&mm_prof)
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            if (!mm_switch_lock_.try_lock())
            {
                return false;
            }
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(mm_prof));
            return true;
        }

#ifdef _MSC_VER
        _Acquires_lock_(mm_switch_lock_)
#endif
            inline void lock(std::unique_ptr<MMProf> &&mm_prof)
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            mm_switch_lock_.lock();
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(mm_prof));
        }

        inline bool try_lock(MMProf *&&mm_prof)
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            if (!mm_switch_lock_.try_lock())
            {
                return false;
            }
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(mm_prof));
            return true;
        }

#ifdef _MSC_VER
        _Acquires_lock_(mm_switch_lock_)
#endif
            inline void lock(MMProf *&&mm_prof)
        {
            if (mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is already owned");
            }
            mm_switch_lock_.lock();
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(mm_prof));
        }

#ifdef _MSC_VER
        _Releases_lock_(mm_switch_lock_)
#endif
            inline void unlock()
        {
            if (!mm_switch_lock_.owns_lock())
            {
                throw std::runtime_error("lock is not owned");
            }
            old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(old_prof_));
            mm_switch_lock_.unlock();
        }

        ~MMProfGuard()
        {
            if (mm_switch_lock_.owns_lock())
            {
                old_prof_ = MemoryManager::SwitchProfileThreadUnsafe(std::move(old_prof_));
                mm_switch_lock_.unlock();
            }
        }

        inline bool owns_lock() noexcept
        {
            return mm_switch_lock_.owns_lock();
        }

    private:
        std::unique_ptr<MMProf> old_prof_;

        std::unique_lock<std::mutex> mm_switch_lock_;
    };
#endif
}
