// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.h"
#include "seal/plaintext.h"
#include "seal/util/defines.h"
#include <vector>
#ifdef SEAL_USE_MSGSL
#include "gsl/span"
#endif

namespace seal
{
     
    class BatchEncoder
    {
    public:
         
        BatchEncoder(const SEALContext &context);

        void encode(const std::vector<std::uint64_t> &values, Plaintext &destination) const;

        void encode(const std::vector<std::int64_t> &values, Plaintext &destination) const;
#ifdef SEAL_USE_MSGSL
         
        void encode(gsl::span<const std::uint64_t> values, Plaintext &destination) const;

        void encode(gsl::span<const std::int64_t> values, Plaintext &destination) const;
#endif
         
        void decode(
            const Plaintext &plain, std::vector<std::uint64_t> &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        void decode(
            const Plaintext &plain, std::vector<std::int64_t> &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;
#ifdef SEAL_USE_MSGSL
         
        void decode(
            const Plaintext &plain, gsl::span<std::uint64_t> destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        void decode(
            const Plaintext &plain, gsl::span<std::int64_t> destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;
#endif
         
        SEAL_NODISCARD inline auto slot_count() const noexcept
        {
            return slots_;
        }

    private:
        BatchEncoder(const BatchEncoder &copy) = delete;

        BatchEncoder(BatchEncoder &&source) = delete;

        BatchEncoder &operator=(const BatchEncoder &assign) = delete;

        BatchEncoder &operator=(BatchEncoder &&assign) = delete;

        void populate_roots_of_unity_vector(const SEALContext::ContextData &context_data);

        void populate_matrix_reps_index_map();

        void reverse_bits(std::uint64_t *input);

        MemoryPoolHandle pool_ = MemoryManager::GetPool();

        SEALContext context_;

        std::size_t slots_;

        util::Pointer<std::uint64_t> roots_of_unity_;

        util::Pointer<std::size_t> matrix_reps_index_map_;
    };
}
