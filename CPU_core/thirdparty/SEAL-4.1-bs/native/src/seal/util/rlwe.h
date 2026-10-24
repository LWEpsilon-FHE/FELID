// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "seal/publickey.h"
#include "seal/randomgen.h"
#include "seal/secretkey.h"
#include <cstdint>

namespace seal
{
    namespace util
    {
         
        void sample_poly_ternary(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);
        
        void sample_poly_sparse_ternary(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void sample_poly_normal(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void sample_poly_cbd(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void sample_poly_uniform(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void sample_poly_uniform_seal_3_4(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void sample_poly_uniform_seal_3_5(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        void encrypt_zero_asymmetric(
            const PublicKey &public_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
            Ciphertext &destination);

        void encrypt_zero_symmetric(
            const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
            bool save_seed, Ciphertext &destination);
    }  
}
