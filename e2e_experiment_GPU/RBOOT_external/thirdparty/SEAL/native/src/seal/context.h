// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/encryptionparams.h"
#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/galois.h"
#include "seal/util/ntt.h"
#include "seal/util/pointer.h"
#include "seal/util/rns.h"
#include <memory>
#include <unordered_map>

namespace seal
{
     
    class EncryptionParameterQualifiers
    {
    public:
         
        enum class error_type : int
        {
             
            none = -1,

            success = 0,

            invalid_scheme = 1,

            invalid_coeff_modulus_size = 2,

            invalid_coeff_modulus_bit_count = 3,

            invalid_coeff_modulus_no_ntt = 4,

            invalid_poly_modulus_degree = 5,

            invalid_poly_modulus_degree_non_power_of_two = 6,

            invalid_parameters_too_large = 7,

            invalid_parameters_insecure = 8,

            failed_creating_rns_base = 9,

            invalid_plain_modulus_bit_count = 10,

            invalid_plain_modulus_coprimality = 11,

            invalid_plain_modulus_too_large = 12,

            invalid_plain_modulus_nonzero = 13,

            failed_creating_rns_tool = 14,
        };

        error_type parameter_error;

        SEAL_NODISCARD const char *parameter_error_name() const noexcept;

        SEAL_NODISCARD const char *parameter_error_message() const noexcept;

        SEAL_NODISCARD inline bool parameters_set() const noexcept
        {
            return parameter_error == error_type::success;
        }

        bool using_fft;

        bool using_ntt;

        bool using_batching;

        bool using_fast_plain_lift;

        bool using_descending_modulus_chain;

        sec_level_type sec_level;

    private:
        EncryptionParameterQualifiers()
            : parameter_error(error_type::none), using_fft(false), using_ntt(false), using_batching(false),
              using_fast_plain_lift(false), using_descending_modulus_chain(false), sec_level(sec_level_type::none)
        {}

        friend class SEALContext;
    };

    class SEALContext
    {
    public:
         
        class ContextData
        {
            friend class SEALContext;

        public:
            ContextData() = delete;

            ContextData(const ContextData &copy) = delete;

            ContextData(ContextData &&move) = default;

            ContextData &operator=(ContextData &&move) = default;

            SEAL_NODISCARD inline auto &parms() const noexcept
            {
                return parms_;
            }

            SEAL_NODISCARD inline auto &parms_id() const noexcept
            {
                return parms_.parms_id();
            }

            SEAL_NODISCARD inline auto qualifiers() const noexcept
            {
                return qualifiers_;
            }

            SEAL_NODISCARD inline auto total_coeff_modulus() const noexcept
            {
                return total_coeff_modulus_.get();
            }

            SEAL_NODISCARD inline int total_coeff_modulus_bit_count() const noexcept
            {
                return total_coeff_modulus_bit_count_;
            }

            SEAL_NODISCARD inline auto rns_tool() const noexcept
            {
                return rns_tool_.get();
            }

            SEAL_NODISCARD inline auto small_ntt_tables() const noexcept
            {
                return small_ntt_tables_.get();
            }

            SEAL_NODISCARD inline auto plain_ntt_tables() const noexcept
            {
                return plain_ntt_tables_.get();
            }

            SEAL_NODISCARD inline auto galois_tool() const noexcept
            {
                return galois_tool_.get();
            }

            SEAL_NODISCARD inline auto coeff_div_plain_modulus() const noexcept
            {
                return coeff_div_plain_modulus_.get();
            }

            SEAL_NODISCARD inline auto plain_upper_half_threshold() const noexcept
            {
                return plain_upper_half_threshold_;
            }

            SEAL_NODISCARD inline auto plain_upper_half_increment() const noexcept
            {
                return plain_upper_half_increment_.get();
            }

            SEAL_NODISCARD inline auto upper_half_threshold() const noexcept
            {
                return upper_half_threshold_.get();
            }

            SEAL_NODISCARD inline auto upper_half_increment() const noexcept
            {
                return upper_half_increment_.get();
            }

            SEAL_NODISCARD inline auto coeff_modulus_mod_plain_modulus() const noexcept -> std::uint64_t
            {
                return coeff_modulus_mod_plain_modulus_;
            }

            SEAL_NODISCARD inline auto prev_context_data() const noexcept
            {
                return prev_context_data_.lock();
            }

            SEAL_NODISCARD inline auto next_context_data() const noexcept
            {
                return next_context_data_;
            }

            SEAL_NODISCARD inline std::size_t chain_index() const noexcept
            {
                return chain_index_;
            }

        private:
            ContextData(EncryptionParameters parms, MemoryPoolHandle pool) : pool_(std::move(pool)), parms_(parms)
            {
                if (!pool_)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }
            }

            MemoryPoolHandle pool_;

            EncryptionParameters parms_;

            EncryptionParameterQualifiers qualifiers_;

            util::Pointer<util::RNSTool> rns_tool_;

            util::Pointer<util::NTTTables> small_ntt_tables_;

            util::Pointer<util::NTTTables> plain_ntt_tables_;

            util::Pointer<util::GaloisTool> galois_tool_;

            util::Pointer<std::uint64_t> total_coeff_modulus_;

            int total_coeff_modulus_bit_count_ = 0;

            util::Pointer<util::MultiplyUIntModOperand> coeff_div_plain_modulus_;

            std::uint64_t plain_upper_half_threshold_ = 0;

            util::Pointer<std::uint64_t> plain_upper_half_increment_;

            util::Pointer<std::uint64_t> upper_half_threshold_;

            util::Pointer<std::uint64_t> upper_half_increment_;

            std::uint64_t coeff_modulus_mod_plain_modulus_ = 0;

            std::weak_ptr<const ContextData> prev_context_data_;

            std::shared_ptr<const ContextData> next_context_data_{ nullptr };

            std::size_t chain_index_ = 0;
        };

        SEALContext(
            const EncryptionParameters &parms, bool expand_mod_chain = true,
            sec_level_type sec_level = sec_level_type::tc128)
            : SEALContext(parms, expand_mod_chain, sec_level, MemoryManager::GetPool())
        {}

        SEALContext(const SEALContext &copy) = default;

        SEALContext(SEALContext &&source) = default;

        SEALContext &operator=(const SEALContext &assign) = default;

        SEALContext &operator=(SEALContext &&assign) = default;

        SEAL_NODISCARD inline auto get_context_data(parms_id_type parms_id) const
        {
            auto data = context_data_map_.find(parms_id);
            return (data != context_data_map_.end()) ? data->second : std::shared_ptr<ContextData>{ nullptr };
        }

        SEAL_NODISCARD inline auto key_context_data() const
        {
            auto data = context_data_map_.find(key_parms_id_);
            return (data != context_data_map_.end()) ? data->second : std::shared_ptr<ContextData>{ nullptr };
        }

        SEAL_NODISCARD inline auto first_context_data() const
        {
            auto data = context_data_map_.find(first_parms_id_);
            return (data != context_data_map_.end()) ? data->second : std::shared_ptr<ContextData>{ nullptr };
        }

        SEAL_NODISCARD inline auto last_context_data() const
        {
            auto data = context_data_map_.find(last_parms_id_);
            return (data != context_data_map_.end()) ? data->second : std::shared_ptr<ContextData>{ nullptr };
        }

        SEAL_NODISCARD inline auto parameters_set() const
        {
            return first_context_data() ? first_context_data()->qualifiers_.parameters_set() : false;
        }

        SEAL_NODISCARD inline const char *parameter_error_name() const
        {
            return first_context_data() ? first_context_data()->qualifiers_.parameter_error_name()
                                        : "SEALContext is empty";
        }

        SEAL_NODISCARD inline const char *parameter_error_message() const
        {
            return first_context_data() ? first_context_data()->qualifiers_.parameter_error_message()
                                        : "SEALContext is empty";
        }

        SEAL_NODISCARD inline auto &key_parms_id() const noexcept
        {
            return key_parms_id_;
        }

        SEAL_NODISCARD inline auto &first_parms_id() const noexcept
        {
            return first_parms_id_;
        }

        SEAL_NODISCARD inline auto &last_parms_id() const noexcept
        {
            return last_parms_id_;
        }

        SEAL_NODISCARD inline bool using_keyswitching() const noexcept
        {
            return using_keyswitching_;
        }

    private:
         
        SEALContext(EncryptionParameters parms, bool expand_mod_chain, sec_level_type sec_level, MemoryPoolHandle pool);

        ContextData validate(EncryptionParameters parms);

        parms_id_type create_next_context_data(const parms_id_type &prev_parms);

        MemoryPoolHandle pool_;

        parms_id_type key_parms_id_;

        parms_id_type first_parms_id_;

        parms_id_type last_parms_id_;

        std::unordered_map<parms_id_type, std::shared_ptr<const ContextData>> context_data_map_{};

        sec_level_type sec_level_;

        bool using_keyswitching_;
    };
}
