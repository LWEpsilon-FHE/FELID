// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/plaintext.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/serializable.h"
#include "seal/util/defines.h"
#include "seal/util/ntt.h"
#include <vector>

namespace seal
{
     
    class Encryptor
    {
    public:
         
        Encryptor(const SEALContext &context, const PublicKey &public_key);

        Encryptor(const SEALContext &context, const SecretKey &secret_key);

        Encryptor(const SEALContext &context, const PublicKey &public_key, const SecretKey &secret_key);

        inline void set_public_key(const PublicKey &public_key)
        {
            if (!is_valid_for(public_key, context_))
            {
                throw std::invalid_argument("public key is not valid for encryption parameters");
            }
            public_key_ = public_key;
        }

        inline void set_secret_key(const SecretKey &secret_key)
        {
            if (!is_valid_for(secret_key, context_))
            {
                throw std::invalid_argument("secret key is not valid for encryption parameters");
            }
            secret_key_ = secret_key;
        }

        inline void encrypt(
            const Plaintext &plain, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_internal(plain, true, false, destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt(
            const Plaintext &plain, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            Ciphertext destination;
            encrypt_internal(plain, true, true, destination, pool);
            return destination;
        }

        inline void encrypt_zero(Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_zero(context_.first_parms_id(), destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt_zero(
            parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            Ciphertext destination;
            encrypt_zero_internal(parms_id, true, true, destination, pool);
            return destination;
        }

        inline void encrypt_zero(
            parms_id_type parms_id, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_zero_internal(parms_id, true, false, destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt_zero(
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            return encrypt_zero(context_.first_parms_id(), pool);
        }

        inline void encrypt_symmetric(
            const Plaintext &plain, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_internal(plain, false, false, destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt_symmetric(
            const Plaintext &plain, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            Ciphertext destination;
            encrypt_internal(plain, false, true, destination, pool);
            return destination;
        }

        inline void encrypt_zero_symmetric(
            parms_id_type parms_id, Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_zero_internal(parms_id, false, false, destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt_zero_symmetric(
            parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            Ciphertext destination;
            encrypt_zero_internal(parms_id, false, true, destination, pool);
            return destination;
        }

        inline void encrypt_zero_symmetric(
            Ciphertext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encrypt_zero_symmetric(context_.first_parms_id(), destination, pool);
        }

        SEAL_NODISCARD inline Serializable<Ciphertext> encrypt_zero_symmetric(
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            return encrypt_zero_symmetric(context_.first_parms_id(), pool);
        }

        struct EncryptorPrivateHelper;

    private:
        Encryptor(const Encryptor &copy) = delete;

        Encryptor(Encryptor &&source) = delete;

        Encryptor &operator=(const Encryptor &assign) = delete;

        Encryptor &operator=(Encryptor &&assign) = delete;

        void encrypt_zero_internal(
            parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        void encrypt_internal(
            const Plaintext &plain, bool is_asymmetric, bool save_seed, Ciphertext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const;

        SEALContext context_;

        PublicKey public_key_;

        SecretKey secret_key_;
    };
}
