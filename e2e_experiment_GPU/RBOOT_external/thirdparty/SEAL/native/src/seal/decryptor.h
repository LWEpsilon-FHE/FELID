// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/plaintext.h"
#include "seal/randomgen.h"
#include "seal/secretkey.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/locks.h"
#include "seal/util/ntt.h"
#include "seal/util/rns.h"

namespace seal
{
     
    class Decryptor
    {
    public:
         
        Decryptor(const SEALContext &context, const SecretKey &secret_key);

        void decrypt(const Ciphertext &encrypted, Plaintext &destination);

        SEAL_NODISCARD int invariant_noise_budget(const Ciphertext &encrypted);

    private:
        void bfv_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool);

        void ckks_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool);

        Decryptor(const Decryptor &copy) = delete;

        Decryptor(Decryptor &&source) = delete;

        Decryptor &operator=(const Decryptor &assign) = delete;

        Decryptor &operator=(Decryptor &&assign) = delete;

        void compute_secret_key_array(std::size_t max_power);

        void dot_product_ct_sk_array(const Ciphertext &encrypted, util::RNSIter destination, MemoryPoolHandle pool);

        MemoryPoolHandle pool_ = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

        SEALContext context_;

        std::size_t secret_key_array_size_ = 0;

        util::Pointer<std::uint64_t> secret_key_array_;

        mutable util::ReaderWriterLocker secret_key_array_locker_;
    };
}
