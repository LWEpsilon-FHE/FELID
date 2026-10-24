// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/common.h"
#include "seal/util/mempool.h"
#include "seal/util/uintarith.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace std;

namespace seal
{
    namespace util
    {

        constexpr double MemoryPool::alloc_size_multiplier;

        constexpr size_t MemoryPool::max_pool_head_count;

        constexpr size_t MemoryPool::first_alloc_count;

        MemoryPoolHeadMT::MemoryPoolHeadMT(size_t item_byte_count, bool clear_on_destruction)
            : clear_on_destruction_(clear_on_destruction), locked_(false), item_byte_count_(item_byte_count),
              item_count_(MemoryPool::first_alloc_count), first_item_(nullptr)
        {
            if ((item_byte_count_ == 0) || (item_byte_count_ > MemoryPool::max_batch_alloc_byte_count) ||
                (mul_safe(item_byte_count_, MemoryPool::first_alloc_count) > MemoryPool::max_batch_alloc_byte_count))
            {
                throw invalid_argument("invalid allocation size");
            }

            allocation new_alloc;
            try
            {
                new_alloc.data_ptr = SEAL_MALLOC(mul_safe(MemoryPool::first_alloc_count, item_byte_count_));
            }
            catch (const bad_alloc &)
            {
                 
                throw;
            }

            new_alloc.size = MemoryPool::first_alloc_count;
            new_alloc.free = MemoryPool::first_alloc_count;
            new_alloc.head_ptr = new_alloc.data_ptr;
            allocs_.clear();
            allocs_.push_back(new_alloc);
        }

        MemoryPoolHeadMT::~MemoryPoolHeadMT() noexcept
        {
            bool expected = false;
            while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire))
            {
                expected = false;
            }

            MemoryPoolItem *curr_item = first_item_;
            while (curr_item)
            {
                MemoryPoolItem *next_item = curr_item->next();
                delete curr_item;
                curr_item = next_item;
            }
            first_item_ = nullptr;

            if (clear_on_destruction_)
            {
                 
                for (auto &alloc : allocs_)
                {
                    size_t curr_alloc_byte_count = mul_safe(item_byte_count_, alloc.size);
                    seal_memzero(alloc.data_ptr, curr_alloc_byte_count);

                    SEAL_FREE(alloc.data_ptr);
                }
            }
            else
            {
                 
                for (auto &alloc : allocs_)
                {
                     
                    SEAL_FREE(alloc.data_ptr);
                }
            }

            allocs_.clear();
        }

        MemoryPoolItem *MemoryPoolHeadMT::get()
        {
            bool expected = false;
            while (!locked_.compare_exchange_strong(expected, true, memory_order_acquire))
            {
                expected = false;
            }
            MemoryPoolItem *old_first = first_item_;

            if (old_first == nullptr)
            {
                allocation &last_alloc = allocs_.back();
                MemoryPoolItem *new_item = nullptr;
                if (last_alloc.free > 0)
                {
                     
                    new_item = new MemoryPoolItem(last_alloc.head_ptr);
                    last_alloc.free--;
                    last_alloc.head_ptr += item_byte_count_;
                }
                else
                {
                     
                    allocation new_alloc;

                    size_t new_size = safe_cast<size_t>(
                        ceil(MemoryPool::alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                    size_t new_alloc_byte_count = mul_safe(new_size, item_byte_count_);
                    if (new_alloc_byte_count > MemoryPool::max_batch_alloc_byte_count)
                    {
                        new_size = last_alloc.size;
                        new_alloc_byte_count = new_size * item_byte_count_;
                    }

                    try
                    {
                        new_alloc.data_ptr = SEAL_MALLOC(new_alloc_byte_count);
                    }
                    catch (const bad_alloc &)
                    {
                         
                        throw;
                    }

                    new_alloc.size = new_size;
                    new_alloc.free = new_size - 1;
                    new_alloc.head_ptr = new_alloc.data_ptr + item_byte_count_;
                    allocs_.push_back(new_alloc);
                    item_count_ += new_size;
                    new_item = new MemoryPoolItem(new_alloc.data_ptr);
                }

                locked_.store(false, memory_order_release);
                return new_item;
            }

            first_item_ = old_first->next();
            old_first->next() = nullptr;
            locked_.store(false, memory_order_release);
            return old_first;
        }

        MemoryPoolHeadST::MemoryPoolHeadST(size_t item_byte_count, bool clear_on_destruction)
            : clear_on_destruction_(clear_on_destruction), item_byte_count_(item_byte_count),
              item_count_(MemoryPool::first_alloc_count), first_item_(nullptr)
        {
            if ((item_byte_count_ == 0) || (item_byte_count_ > MemoryPool::max_batch_alloc_byte_count) ||
                (mul_safe(item_byte_count_, MemoryPool::first_alloc_count) > MemoryPool::max_batch_alloc_byte_count))
            {
                throw invalid_argument("invalid allocation size");
            }

            allocation new_alloc;
            try
            {
                new_alloc.data_ptr = SEAL_MALLOC(mul_safe(MemoryPool::first_alloc_count, item_byte_count_));
            }
            catch (const bad_alloc &)
            {
                 
                throw;
            }

            new_alloc.size = MemoryPool::first_alloc_count;
            new_alloc.free = MemoryPool::first_alloc_count;
            new_alloc.head_ptr = new_alloc.data_ptr;
            allocs_.clear();
            allocs_.push_back(new_alloc);
        }

        MemoryPoolHeadST::~MemoryPoolHeadST() noexcept
        {
             
            MemoryPoolItem *curr_item = first_item_;
            while (curr_item)
            {
                MemoryPoolItem *next_item = curr_item->next();
                delete curr_item;
                curr_item = next_item;
            }
            first_item_ = nullptr;

            if (clear_on_destruction_)
            {
                 
                for (auto &alloc : allocs_)
                {
                    size_t curr_alloc_byte_count = mul_safe(item_byte_count_, alloc.size);
                    seal_memzero(alloc.data_ptr, curr_alloc_byte_count);

                    SEAL_FREE(alloc.data_ptr);
                }
            }
            else
            {
                 
                for (auto &alloc : allocs_)
                {
                     
                    SEAL_FREE(alloc.data_ptr);
                }
            }

            allocs_.clear();
        }

        MemoryPoolItem *MemoryPoolHeadST::get()
        {
            MemoryPoolItem *old_first = first_item_;

            if (old_first == nullptr)
            {
                allocation &last_alloc = allocs_.back();
                MemoryPoolItem *new_item = nullptr;
                if (last_alloc.free > 0)
                {
                     
                    new_item = new MemoryPoolItem(last_alloc.head_ptr);
                    last_alloc.free--;
                    last_alloc.head_ptr += item_byte_count_;
                }
                else
                {
                     
                    allocation new_alloc;

                    size_t new_size = safe_cast<size_t>(
                        ceil(MemoryPool::alloc_size_multiplier * static_cast<double>(last_alloc.size)));
                    size_t new_alloc_byte_count = mul_safe(new_size, item_byte_count_);
                    if (new_alloc_byte_count > MemoryPool::max_batch_alloc_byte_count)
                    {
                        new_size = last_alloc.size;
                        new_alloc_byte_count = new_size * item_byte_count_;
                    }

                    try
                    {
                        new_alloc.data_ptr = SEAL_MALLOC(new_alloc_byte_count);
                    }
                    catch (const bad_alloc &)
                    {
                         
                        throw;
                    }

                    new_alloc.size = new_size;
                    new_alloc.free = new_size - 1;
                    new_alloc.head_ptr = new_alloc.data_ptr + item_byte_count_;
                    allocs_.push_back(new_alloc);
                    item_count_ += new_size;
                    new_item = new MemoryPoolItem(new_alloc.data_ptr);
                }

                return new_item;
            }

            first_item_ = old_first->next();
            old_first->next() = nullptr;
            return old_first;
        }

        const size_t MemoryPool::max_single_alloc_byte_count = []() -> size_t {
            int bit_shift = static_cast<int>(ceil(log2(MemoryPool::alloc_size_multiplier)));
            if (bit_shift < 0 || unsigned_geq(bit_shift, sizeof(size_t) * static_cast<size_t>(bits_per_byte)))
            {
                throw logic_error("alloc_size_multiplier too large");
            }
            return numeric_limits<size_t>::max() >> bit_shift;
        }();

        const size_t MemoryPool::max_batch_alloc_byte_count = []() -> size_t {
            int bit_shift = static_cast<int>(ceil(log2(MemoryPool::alloc_size_multiplier)));
            if (bit_shift < 0 || unsigned_geq(bit_shift, sizeof(size_t) * static_cast<size_t>(bits_per_byte)))
            {
                throw logic_error("alloc_size_multiplier too large");
            }
            return numeric_limits<size_t>::max() >> bit_shift;
        }();

        MemoryPoolMT::~MemoryPoolMT() noexcept
        {
            WriterLock lock(pools_locker_.acquire_write());
            for (MemoryPoolHead *head : pools_)
            {
                delete head;
            }
            pools_.clear();
        }

        Pointer<seal_byte> MemoryPoolMT::get_for_byte_count(size_t byte_count)
        {
            if (byte_count > max_single_alloc_byte_count)
            {
                throw invalid_argument("invalid allocation size");
            }
            else if (byte_count == 0)
            {
                return Pointer<seal_byte>();
            }

            ReaderLock reader_lock(pools_locker_.acquire_read());
            size_t start = 0;
            size_t end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                size_t mid_byte_count = mid_head->item_byte_count();
                if (byte_count < mid_byte_count)
                {
                    start = mid + 1;
                }
                else if (byte_count > mid_byte_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer<seal_byte>(mid_head);
                }
            }
            reader_lock.unlock();

            WriterLock writer_lock(pools_locker_.acquire_write());
            start = 0;
            end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                size_t mid_byte_count = mid_head->item_byte_count();
                if (byte_count < mid_byte_count)
                {
                    start = mid + 1;
                }
                else if (byte_count > mid_byte_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer<seal_byte>(mid_head);
                }
            }

            if (pools_.size() >= max_pool_head_count)
            {
                throw runtime_error("maximum pool head count reached");
            }

            MemoryPoolHead *new_head = new MemoryPoolHeadMT(byte_count, clear_on_destruction_);
            if (!pools_.empty())
            {
                pools_.insert(pools_.begin() + static_cast<ptrdiff_t>(start), new_head);
            }
            else
            {
                pools_.emplace_back(new_head);
            }

            return Pointer<seal_byte>(new_head);
        }

        size_t MemoryPoolMT::alloc_byte_count() const
        {
            ReaderLock lock(pools_locker_.acquire_read());

            return accumulate(pools_.cbegin(), pools_.cend(), size_t(0), [](size_t byte_count, MemoryPoolHead *head) {
                return add_safe(byte_count, mul_safe(head->item_count(), head->item_byte_count()));
            });
        }

        MemoryPoolST::~MemoryPoolST() noexcept
        {
            for (MemoryPoolHead *head : pools_)
            {
                delete head;
            }
            pools_.clear();
        }

        Pointer<seal_byte> MemoryPoolST::get_for_byte_count(size_t byte_count)
        {
            if (byte_count > MemoryPool::max_single_alloc_byte_count)
            {
                throw invalid_argument("invalid allocation size");
            }
            else if (byte_count == 0)
            {
                return Pointer<seal_byte>();
            }

            size_t start = 0;
            size_t end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                size_t mid_byte_count = mid_head->item_byte_count();
                if (byte_count < mid_byte_count)
                {
                    start = mid + 1;
                }
                else if (byte_count > mid_byte_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer<seal_byte>(mid_head);
                }
            }

            if (pools_.size() >= max_pool_head_count)
            {
                throw runtime_error("maximum pool head count reached");
            }

            MemoryPoolHead *new_head = new MemoryPoolHeadST(byte_count, clear_on_destruction_);
            if (!pools_.empty())
            {
                pools_.insert(pools_.begin() + static_cast<ptrdiff_t>(start), new_head);
            }
            else
            {
                pools_.emplace_back(new_head);
            }

            return Pointer<seal_byte>(new_head);
        }

        size_t MemoryPoolST::alloc_byte_count() const
        {
            return accumulate(pools_.cbegin(), pools_.cend(), size_t(0), [](size_t byte_count, MemoryPoolHead *head) {
                return add_safe(byte_count, mul_safe(head->item_count(), head->item_byte_count()));
            });
        }
    }  
}
