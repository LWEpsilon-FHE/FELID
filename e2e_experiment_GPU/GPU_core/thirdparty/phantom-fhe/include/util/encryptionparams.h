#pragma once

#include "modulus.h"
#include "hash.h"
#include "defines.h"
#include "globals.h"
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

namespace phantom {
     
    enum class scheme_type : std::uint8_t {
        none = 0,
         
        bgv = 1,
         
        bfv = 2,
         
        ckks = 3  
    };

    enum class mul_tech_type : std::uint8_t {
        none = 0,
         
        behz = 1,
         
        hps = 2,
         
        hps_overq = 3,
         
        hps_overq_leveled = 4  
    };

    using parms_id_type = util::HashFunction::hash_block_type;

    constexpr parms_id_type parms_id_zero = util::HashFunction::hash_zero_block;

    class EncryptionParameters {

    public:
         
        explicit EncryptionParameters(scheme_type scheme = scheme_type::none) : scheme_(scheme) {
             
            if (scheme_ == scheme_type::bfv)
                mul_tech_ = mul_tech_type::hps;
            else
                mul_tech_ = mul_tech_type::none;
            compute_parms_id();
        }

        EncryptionParameters(const EncryptionParameters &copy) = default;

        EncryptionParameters &operator=(const EncryptionParameters &assign) = default;

        EncryptionParameters(EncryptionParameters &&source) = default;

        EncryptionParameters &operator=(EncryptionParameters &&assign) = default;

        inline void set_mul_tech(mul_tech_type mul_tech) {
            if (scheme_ == scheme_type::bfv) {
                if (is_valid_mul_tech(mul_tech)) {
                    mul_tech_ = mul_tech;
                }
                else {
                    throw std::invalid_argument("unsupported multiplication technique for BFV");
                }
            }
            else {
                throw std::invalid_argument("mul_tech selection is only supported for BFV");
            }
        }

        inline void set_poly_modulus_degree(std::size_t poly_modulus_degree) {
            if (scheme_ == scheme_type::none && poly_modulus_degree) {
                throw std::logic_error("poly_modulus_degree is not supported for this scheme");
            }

            poly_modulus_degree_ = poly_modulus_degree;

            compute_parms_id();
        }

        inline void set_special_modulus_size(std::size_t special_modulus_size) {
            if (scheme_ == scheme_type::none && special_modulus_size) {
                throw std::logic_error("special_modulus_size is not supported for this scheme");
            }

            special_modulus_size_ = special_modulus_size;

            compute_parms_id();
        }

        inline void set_secret_key_hamming_weight(std::size_t secret_key_hamming_weight)
        {
            if (scheme_ == scheme_type::none && secret_key_hamming_weight)
            {
                throw std::logic_error("secret key hamming weight is not supported for this scheme");
            }

            secret_key_hamming_weight_ = secret_key_hamming_weight;

            compute_parms_id();
        }

        inline void set_sparse_slots(std::size_t sparse_slots)
        {
            if (scheme_ == scheme_type::none && sparse_slots)
            {
                throw std::logic_error("secret key hamming weight is not supported for this scheme");
            }

            if ((sparse_slots & (sparse_slots - 1)) != 0)
            {
                throw std::logic_error("secret key hamming weight is not zero or power-of-two");
            }

            sparse_slots_= sparse_slots;

            compute_parms_id();
        }

        inline void set_galois_elts(const std::vector<uint32_t> &galois_elts) {
            galois_elts_ = galois_elts;
        }

        inline void set_coeff_modulus(const std::vector<arith::Modulus> &coeff_modulus) {
             
            if (scheme_ == scheme_type::none) {
                if (!coeff_modulus.empty()) {
                    throw std::logic_error("coeff_modulus is not supported for this scheme");
                }
            }
            else if (coeff_modulus.size() > COEFF_MOD_COUNT_MAX ||
                     coeff_modulus.size() < COEFF_MOD_COUNT_MIN) {
                throw std::invalid_argument("coeff_modulus is invalid");
            }

            key_modulus_ = coeff_modulus;

            coeff_modulus_ = coeff_modulus;

            compute_parms_id();
        }

        inline void set_plain_modulus(const arith::Modulus &plain_modulus) {
             
            if (scheme_ != scheme_type::bfv && scheme_ != scheme_type::bgv && !plain_modulus.is_zero()) {
                throw std::logic_error("plain_modulus is not supported for this scheme");
            }

            plain_modulus_ = plain_modulus;

            compute_parms_id();
        }

        [[nodiscard]] inline scheme_type scheme() const noexcept {
            return scheme_;
        }

        [[nodiscard]] inline mul_tech_type mul_tech() const noexcept {
            return mul_tech_;
        }

        [[nodiscard]] inline std::size_t poly_modulus_degree() const noexcept {
            return poly_modulus_degree_;
        }

        [[nodiscard]] inline std::size_t special_modulus_size() const noexcept {
            return special_modulus_size_;
        }

        [[nodiscard]] inline std::size_t secret_key_hamming_weight() const noexcept {
            return secret_key_hamming_weight_;
        }

        [[nodiscard]] inline std::size_t sparse_slots() const noexcept {
            return sparse_slots_;
        }

        [[nodiscard]] inline auto galois_elts() const noexcept {
            return galois_elts_;
        }

        [[nodiscard]] inline const std::vector<arith::Modulus> &key_modulus() const noexcept {
            return key_modulus_;
        }

        [[nodiscard]] inline auto coeff_modulus() const noexcept -> const std::vector<arith::Modulus> & {
            return coeff_modulus_;
        }

        [[nodiscard]] inline auto coeff_modulus() noexcept -> std::vector<arith::Modulus> & {
            return coeff_modulus_;
        }

        [[nodiscard]] inline const arith::Modulus &plain_modulus() const noexcept {
            return plain_modulus_;
        }

        [[nodiscard]] inline bool operator==(const EncryptionParameters &other) const noexcept {
            return (parms_id_ == other.parms_id_);
        }

        [[nodiscard]] inline bool operator!=(const EncryptionParameters &other) const noexcept {
            return (parms_id_ != other.parms_id_);
        }

        [[nodiscard]] inline auto &parms_id() const noexcept {
            return parms_id_;
        }

        inline void update_parms_id() {
            compute_parms_id();
        }

    private:
        [[nodiscard]] static bool is_valid_scheme(std::uint8_t scheme) noexcept {
            switch (scheme) {
                case static_cast<std::uint8_t>(scheme_type::none):
                case static_cast<std::uint8_t>(scheme_type::bfv):
                case static_cast<std::uint8_t>(scheme_type::ckks):
                case static_cast<std::uint8_t>(scheme_type::bgv):
                    return true;

                default:
                    return false;
            }
        }

        [[nodiscard]] static bool is_valid_mul_tech(mul_tech_type mul_tech) noexcept {
            switch (mul_tech) {
                case mul_tech_type::behz:
                case mul_tech_type::hps:
                case mul_tech_type::hps_overq:
                case mul_tech_type::hps_overq_leveled:
                    return true;

                default:
                    return false;
            }
        }

        inline void compute_parms_id() {
            size_t coeff_modulus_size = coeff_modulus_.size();
            size_t key_modulus_size = key_modulus_.size();

            size_t total_uint64_count = size_t(1) +  
                                        size_t(1) +  
                                        size_t(1) +  
                                        key_modulus_size +  
                                        coeff_modulus_size + plain_modulus_.uint64_count();

            std::vector<uint64_t> param_data;
            param_data.resize(total_uint64_count);
            uint64_t *param_data_ptr = param_data.data();

            *param_data_ptr++ = static_cast<uint64_t>(scheme_);

            *param_data_ptr++ = static_cast<uint64_t>(poly_modulus_degree_);

            *param_data_ptr++ = static_cast<uint64_t>(special_modulus_size_);

            for (const auto &mod: key_modulus_) {
                *param_data_ptr++ = mod.value();
            }

            for (const auto &mod: coeff_modulus_) {
                *param_data_ptr++ = mod.value();
            }

            if (plain_modulus_.uint64_count() == 1) {
                *param_data_ptr++ = plain_modulus_.value();
            }

            phantom::util::HashFunction::hash(param_data.data(), total_uint64_count, parms_id_);

            if (parms_id_ == parms_id_zero) {
                throw std::logic_error("parms_id cannot be zero");
            }
        }

        scheme_type scheme_;

        mul_tech_type mul_tech_;

        std::size_t poly_modulus_degree_ = 0;

        std::size_t special_modulus_size_ = 1;

        std::size_t secret_key_hamming_weight_ = 0;

        std::size_t sparse_slots_ = 0;

        std::vector<arith::Modulus> key_modulus_{};

        std::vector<arith::Modulus> coeff_modulus_{};

        std::vector<uint32_t> galois_elts_{};

        arith::Modulus plain_modulus_{};

        parms_id_type parms_id_ = parms_id_zero;
    };
}
