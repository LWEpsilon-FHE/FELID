// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.h"
#include "seal/galoiskeys.h"
#include "seal/memorymanager.h"
#include "seal/publickey.h"
#include "seal/relinkeys.h"
#include "seal/secretkey.h"
#include "seal/serializable.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include <random>

namespace seal
{
     
    class KeyGenerator
    {
    public:
         
        KeyGenerator(const SEALContext &context);

        KeyGenerator(const SEALContext &context, const SecretKey &secret_key);

        SEAL_NODISCARD const SecretKey &secret_key() const;

        inline void create_public_key(PublicKey &destination) const
        {
            destination = generate_pk(false);
        }

        SEAL_NODISCARD inline Serializable<PublicKey> create_public_key() const
        {
            return generate_pk(true);
        }

        inline void create_relin_keys(RelinKeys &destination)
        {
            destination = create_relin_keys(1, false);
        }

        SEAL_NODISCARD inline Serializable<RelinKeys> create_relin_keys()
        {
            return create_relin_keys(1, true);
        }

        inline void create_galois_keys(const std::vector<std::uint32_t> &galois_elts, GaloisKeys &destination)
        {
            destination = create_galois_keys(galois_elts, false);
        }

        SEAL_NODISCARD inline Serializable<GaloisKeys> create_galois_keys(const std::vector<std::uint32_t> &galois_elts)
        {
            return create_galois_keys(galois_elts, true);
        }

        inline void create_galois_keys(const std::vector<int> &steps, GaloisKeys &destination)
        {
            if (!context_.key_context_data()->qualifiers().using_batching)
            {
                throw std::logic_error("encryption parameters do not support batching");
            }
            create_galois_keys(context_.key_context_data()->galois_tool()->get_elts_from_steps(steps), destination);
        }

        SEAL_NODISCARD inline Serializable<GaloisKeys> create_galois_keys(const std::vector<int> &steps)
        {
            if (!context_.key_context_data()->qualifiers().using_batching)
            {
                throw std::logic_error("encryption parameters do not support batching");
            }
            return create_galois_keys(context_.key_context_data()->galois_tool()->get_elts_from_steps(steps));
        }

        inline void create_galois_keys(GaloisKeys &destination)
        {
            create_galois_keys(context_.key_context_data()->galois_tool()->get_elts_all(), destination);
        }

        SEAL_NODISCARD inline Serializable<GaloisKeys> create_galois_keys()
        {
            return create_galois_keys(context_.key_context_data()->galois_tool()->get_elts_all());
        }

        struct KeyGeneratorPrivateHelper;

    private:
        KeyGenerator(const KeyGenerator &copy) = delete;

        KeyGenerator &operator=(const KeyGenerator &assign) = delete;

        KeyGenerator(KeyGenerator &&source) = delete;

        KeyGenerator &operator=(KeyGenerator &&assign) = delete;

        void compute_secret_key_array(const SEALContext::ContextData &context_data, std::size_t max_power);

        void generate_sk(bool is_initialized = false);

        PublicKey generate_pk(bool save_seed) const;

        void generate_kswitch_keys(
            util::ConstPolyIter new_keys, std::size_t num_keys, KSwitchKeys &destination, bool save_seed = false);

        void generate_one_kswitch_key(
            util::ConstRNSIter new_key, std::vector<PublicKey> &destination, bool save_seed = false);

        RelinKeys create_relin_keys(std::size_t count, bool save_seed);

        GaloisKeys create_galois_keys(const std::vector<std::uint32_t> &galois_elts, bool save_seed);

        MemoryPoolHandle pool_ = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

        SEALContext context_;

        SecretKey secret_key_;

        std::size_t secret_key_array_size_ = 0;

        util::Pointer<std::uint64_t> secret_key_array_;

        mutable util::ReaderWriterLocker secret_key_array_locker_;

        bool sk_generated_ = false;
    };
}
