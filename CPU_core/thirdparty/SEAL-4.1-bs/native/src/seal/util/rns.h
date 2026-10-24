// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/iterator.h"
#include "seal/util/ntt.h"
#include "seal/util/pointer.h"
#include "seal/util/uintarithsmallmod.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace seal
{
    namespace util
    {
        class RNSBase
        {
        public:
            RNSBase(const std::vector<Modulus> &rnsbase, MemoryPoolHandle pool);

            RNSBase(RNSBase &&source) = default;

            RNSBase(const RNSBase &copy, MemoryPoolHandle pool);

            RNSBase(const RNSBase &copy) : RNSBase(copy, copy.pool_)
            {}

            RNSBase &operator=(const RNSBase &assign) = delete;

            SEAL_NODISCARD inline const Modulus &operator[](std::size_t index) const
            {
                if (index >= size_)
                {
                    throw std::out_of_range("index is out of range");
                }
                return base_[index];
            }

            SEAL_NODISCARD inline std::size_t size() const noexcept
            {
                return size_;
            }

            SEAL_NODISCARD bool contains(const Modulus &value) const noexcept;

            SEAL_NODISCARD bool is_subbase_of(const RNSBase &superbase) const noexcept;

            SEAL_NODISCARD inline bool is_superbase_of(const RNSBase &subbase) const noexcept
            {
                return subbase.is_subbase_of(*this);
            }

            SEAL_NODISCARD inline bool is_proper_subbase_of(const RNSBase &superbase) const noexcept
            {
                return (size_ < superbase.size_) && is_subbase_of(superbase);
            }

            SEAL_NODISCARD inline bool is_proper_superbase_of(const RNSBase &subbase) const noexcept
            {
                return (size_ > subbase.size_) && !is_subbase_of(subbase);
            }

            SEAL_NODISCARD RNSBase extend(const Modulus &value) const;

            SEAL_NODISCARD RNSBase extend(const RNSBase &other) const;

            SEAL_NODISCARD RNSBase drop() const;

            SEAL_NODISCARD RNSBase drop(const Modulus &value) const;

            void decompose(std::uint64_t *value, MemoryPoolHandle pool) const;

            void decompose_array(std::uint64_t *value, std::size_t count, MemoryPoolHandle pool) const;

            void compose(std::uint64_t *value, MemoryPoolHandle pool) const;

            void compose_array(std::uint64_t *value, std::size_t count, MemoryPoolHandle pool) const;

            SEAL_NODISCARD inline const Modulus *base() const noexcept
            {
                return base_.get();
            }

            SEAL_NODISCARD inline const std::uint64_t *base_prod() const noexcept
            {
                return base_prod_.get();
            }

            SEAL_NODISCARD inline const std::uint64_t *punctured_prod_array() const noexcept
            {
                return punctured_prod_array_.get();
            }

            SEAL_NODISCARD inline const MultiplyUIntModOperand *inv_punctured_prod_mod_base_array() const noexcept
            {
                return inv_punctured_prod_mod_base_array_.get();
            }

        private:
            RNSBase(MemoryPoolHandle pool) : pool_(std::move(pool)), size_(0)
            {
                if (!pool_)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }
            }

            bool initialize();

            MemoryPoolHandle pool_;

            std::size_t size_;

            Pointer<Modulus> base_;

            Pointer<std::uint64_t> base_prod_;

            Pointer<std::uint64_t> punctured_prod_array_;

            Pointer<MultiplyUIntModOperand> inv_punctured_prod_mod_base_array_;
        };

        class BaseConverter
        {
        public:
            BaseConverter(const RNSBase &ibase, const RNSBase &obase, MemoryPoolHandle pool)
                : pool_(std::move(pool)), ibase_(ibase, pool_), obase_(obase, pool_)
            {
                if (!pool_)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }

                initialize();
            }

            SEAL_NODISCARD inline std::size_t ibase_size() const noexcept
            {
                return ibase_.size();
            }

            SEAL_NODISCARD inline std::size_t obase_size() const noexcept
            {
                return obase_.size();
            }

            SEAL_NODISCARD inline const RNSBase &ibase() const noexcept
            {
                return ibase_;
            }

            SEAL_NODISCARD inline const RNSBase &obase() const noexcept
            {
                return obase_;
            }

            void fast_convert(ConstCoeffIter in, CoeffIter out, MemoryPoolHandle pool) const;

            void fast_convert_array(ConstRNSIter in, RNSIter out, MemoryPoolHandle pool) const;

            void exact_convert_array(ConstRNSIter in, CoeffIter out, MemoryPoolHandle) const;

        private:
            BaseConverter(const BaseConverter &copy) = delete;

            BaseConverter(BaseConverter &&source) = delete;

            BaseConverter &operator=(const BaseConverter &assign) = delete;

            BaseConverter &operator=(BaseConverter &&assign) = delete;

            void initialize();

            MemoryPoolHandle pool_;

            RNSBase ibase_;

            RNSBase obase_;

            Pointer<Pointer<std::uint64_t>> base_change_matrix_;
        };

        class RNSTool
        {
        public:
             
            RNSTool(
                std::size_t poly_modulus_degree, const RNSBase &coeff_modulus, const Modulus &plain_modulus,
                MemoryPoolHandle pool);

            void divide_and_round_q_last_inplace(RNSIter input, MemoryPoolHandle pool) const;

            void divide_and_round_q_last_ntt_inplace(
                RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const;

            void fastbconv_sk(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;

            void sm_mrq(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;

            void fast_floor(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;

            void fastbconv_m_tilde(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;

            void decrypt_scale_and_round(ConstRNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const;

            void mod_t_and_divide_q_last_ntt_inplace(
                RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const;

            void decrypt_modt(RNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const;

            SEAL_NODISCARD inline auto inv_q_last_mod_q() const noexcept
            {
                return inv_q_last_mod_q_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk_ntt_tables() const noexcept
            {
                return base_Bsk_ntt_tables_.get();
            }

            SEAL_NODISCARD inline auto base_q() const noexcept
            {
                return base_q_.get();
            }

            SEAL_NODISCARD inline auto base_B() const noexcept
            {
                return base_B_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk() const noexcept
            {
                return base_Bsk_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk_m_tilde() const noexcept
            {
                return base_Bsk_m_tilde_.get();
            }

            SEAL_NODISCARD inline auto base_t_gamma() const noexcept
            {
                return base_t_gamma_.get();
            }

            SEAL_NODISCARD inline auto &m_tilde() const noexcept
            {
                return m_tilde_;
            }

            SEAL_NODISCARD inline auto &m_sk() const noexcept
            {
                return m_sk_;
            }

            SEAL_NODISCARD inline auto &t() const noexcept
            {
                return t_;
            }

            SEAL_NODISCARD inline auto &gamma() const noexcept
            {
                return gamma_;
            }

            SEAL_NODISCARD inline auto &inv_q_last_mod_t() const noexcept
            {
                return inv_q_last_mod_t_;
            }

            SEAL_NODISCARD inline const uint64_t &q_last_mod_t() const noexcept
            {
                return q_last_mod_t_;
            }

        private:
            RNSTool(const RNSTool &copy) = delete;

            RNSTool(RNSTool &&source) = delete;

            RNSTool &operator=(const RNSTool &assign) = delete;

            RNSTool &operator=(RNSTool &&assign) = delete;

            void initialize(std::size_t poly_modulus_degree, const RNSBase &q, const Modulus &t);

            MemoryPoolHandle pool_;

            std::size_t coeff_count_ = 0;

            Pointer<RNSBase> base_q_;

            Pointer<RNSBase> base_B_;

            Pointer<RNSBase> base_Bsk_;

            Pointer<RNSBase> base_Bsk_m_tilde_;

            Pointer<RNSBase> base_t_gamma_;

            Pointer<BaseConverter> base_q_to_Bsk_conv_;

            Pointer<BaseConverter> base_q_to_m_tilde_conv_;

            Pointer<BaseConverter> base_B_to_q_conv_;

            Pointer<BaseConverter> base_B_to_m_sk_conv_;

            Pointer<BaseConverter> base_q_to_t_gamma_conv_;

            Pointer<BaseConverter> base_q_to_t_conv_;

            Pointer<MultiplyUIntModOperand> inv_prod_q_mod_Bsk_;

            MultiplyUIntModOperand neg_inv_prod_q_mod_m_tilde_;

            MultiplyUIntModOperand inv_prod_B_mod_m_sk_;

            MultiplyUIntModOperand inv_gamma_mod_t_;

            Pointer<std::uint64_t> prod_B_mod_q_;

            Pointer<MultiplyUIntModOperand> inv_m_tilde_mod_Bsk_;

            Pointer<std::uint64_t> prod_q_mod_Bsk_;

            Pointer<MultiplyUIntModOperand> neg_inv_q_mod_t_gamma_;

            Pointer<MultiplyUIntModOperand> prod_t_gamma_mod_q_;

            Pointer<MultiplyUIntModOperand> inv_q_last_mod_q_;

            Pointer<NTTTables> base_Bsk_ntt_tables_;

            Modulus m_tilde_;

            Modulus m_sk_;

            Modulus t_;

            Modulus gamma_;

            std::uint64_t inv_q_last_mod_t_ = 1;

            std::uint64_t q_last_mod_t_ = 1;
        };
    }  
}
