#pragma once

#include <memory>

#include "util/encryptionparams.h"
#include "util/modulus.h"
#include "util/ntt.h"
#include "util/rns.h"

#include "galois.cuh"
#include "gputype.h"
#include "rns.cuh"
#include "util.cuh"
#include "cuda_wrapper.cuh"

namespace phantom {

    class ContextData {

    private:

        EncryptionParameters parms_;

        std::shared_ptr<DRNSTool> gpu_rns_tool_;

        std::shared_ptr<arith::RNSNTT> small_ntt_tables_;

        std::shared_ptr<arith::NTT> plain_ntt_tables_;

        std::vector<std::uint64_t> total_coeff_modulus_;

        int total_coeff_modulus_bit_count_ = 0;

        std::vector<uint64_t> coeff_div_plain_modulus_;
        std::vector<uint64_t> coeff_div_plain_modulus_shoup_;

        std::vector<uint64_t> plain_modulus_;
        std::vector<uint64_t> plain_modulus_shoup_;

        std::uint64_t plain_upper_half_threshold_ = 0;

        std::vector<std::uint64_t> plain_upper_half_increment_;

        std::vector<std::uint64_t> upper_half_threshold_;

        std::vector<std::uint64_t> upper_half_increment_;

        std::uint64_t coeff_modulus_mod_plain_modulus_ = 0;

        std::size_t chain_index_ = 0;

    public:
        explicit ContextData(const EncryptionParameters &params, const cudaStream_t &stream);

        ContextData() = delete;

        ContextData(const ContextData &copy) = delete;

        ContextData(ContextData &&move) = default;

        ContextData &operator=(ContextData &&move) = delete;

        [[nodiscard]] auto &parms() const noexcept { return parms_; }

        [[nodiscard]] auto &total_coeff_modulus() const noexcept { return total_coeff_modulus_; }

        [[nodiscard]] int total_coeff_modulus_bit_count() const noexcept { return total_coeff_modulus_bit_count_; }

        [[nodiscard]] auto &gpu_rns_tool() const noexcept { return *gpu_rns_tool_.get(); }

        [[nodiscard]] auto &small_ntt_tables() const noexcept { return small_ntt_tables_; }

        [[nodiscard]] auto &plain_ntt_tables() const noexcept { return plain_ntt_tables_; }

        auto &coeff_div_plain_modulus() const noexcept { return coeff_div_plain_modulus_; }

        auto &coeff_div_plain_modulus_shoup() const noexcept { return coeff_div_plain_modulus_shoup_; }

        auto &plain_modulus() const noexcept { return plain_modulus_; }

        auto &plain_modulus_shoup() const noexcept { return plain_modulus_shoup_; }

        auto plain_upper_half_threshold() const noexcept { return plain_upper_half_threshold_; }

        [[nodiscard]] auto &plain_upper_half_increment() const noexcept { return plain_upper_half_increment_; }

        [[nodiscard]] auto &upper_half_threshold() const noexcept { return upper_half_threshold_; }

        [[nodiscard]] auto &upper_half_increment() const noexcept { return upper_half_increment_; }

        [[nodiscard]] auto coeff_modulus_mod_plain_modulus() const noexcept -> std::uint64_t {
            return coeff_modulus_mod_plain_modulus_;
        }

        [[nodiscard]] std::size_t chain_index() const noexcept { return chain_index_; }

        [[nodiscard]] std::size_t chain_depth() const noexcept { return total_coeff_modulus_.size() - 1; }

        void set_chain_index(const std::size_t chain_index) noexcept { chain_index_ = chain_index; }
    };

}  

class PhantomContext {
public:

    std::vector<phantom::ContextData> context_data_;

    bool using_keyswitching_;

    size_t first_parm_index_;

    phantom::mul_tech_type mul_tech_;

    DNTTTable gpu_rns_tables_;

    DNTTTable gpu_plain_tables_;

    phantom::util::cuda_auto_ptr<uint64_t> coeff_div_plain_;
    phantom::util::cuda_auto_ptr<uint64_t> coeff_div_plain_shoup_;
     
    phantom::util::cuda_auto_ptr<uint64_t> plain_modulus_;  
    phantom::util::cuda_auto_ptr<uint64_t> plain_modulus_shoup_;  

    phantom::util::cuda_auto_ptr<uint64_t> plain_upper_half_increment_;

    std::size_t coeff_mod_size_ = 0;  
    std::size_t poly_degree_ = 0;  
    std::unique_ptr<PhantomGaloisTool> key_galois_tool_;

    explicit PhantomContext(const phantom::EncryptionParameters &params);

    void reinitialize(const phantom::EncryptionParameters &params);

    PhantomContext(const PhantomContext &) = delete;

    void operator=(const PhantomContext &) = delete;

    ~PhantomContext() = default;

    [[nodiscard]] const phantom::ContextData &get_context_data(const size_t index) const {
        if (index >= context_data_.size())
            throw std::invalid_argument("index is invalid!");
        return context_data_[index];
    }

    [[nodiscard]] auto &key_context_data() const {
        auto context_data_size = context_data_.size();
        if (context_data_size == 0)
            throw std::invalid_argument("context_data is null!");
        return context_data_[0];
    }

    [[nodiscard]] auto &first_context_data() const {
        auto context_data_size = context_data_.size();
        if (context_data_size == 0) {
            throw std::invalid_argument("context_data is null!");
        }
        return context_data_[static_cast<size_t>(1)];
    }

    [[nodiscard]] auto &last_context_data() const {
        auto context_data_size = context_data_.size();
        if (context_data_size == 0) {
            throw std::invalid_argument("context_data is null!");
        }
        return context_data_[context_data_size - 1];
    }

    [[nodiscard]] auto &get_context_data_rns_tool(size_t index) const {
        if (index < context_data_.size())
            return context_data_[index].gpu_rns_tool();
        throw std::invalid_argument("index is invalid!!!");
    }

    [[nodiscard]] size_t previous_parm_index(size_t index) const {
        if (index >= context_data_.size())
            throw std::invalid_argument("index not valid");
        if (index < 1)
            return 0;
        return index - 1;
    }

    [[nodiscard]] size_t next_parm_index(size_t index) const {
        if (index >= (context_data_.size() - 1))
            throw std::invalid_argument("index not valid");
        return index + 1;
    }

    [[nodiscard]] auto total_parm_size() const { return context_data_.size(); }

    [[nodiscard]] auto using_keyswitching() const { return using_keyswitching_; }

    [[nodiscard]] auto mul_tech() const { return mul_tech_; }

    [[nodiscard]] auto get_first_index() const { return first_parm_index_; }

    [[nodiscard]] auto get_previous_index(const size_t index) const { return previous_parm_index(index); }

    [[nodiscard]] auto get_next_index(const size_t index) const { return next_parm_index(index); }

    [[nodiscard]] const DNTTTable &gpu_plain_tables() const noexcept { return gpu_plain_tables_; }

    DNTTTable &gpu_plain_tables() { return gpu_plain_tables_; }

    [[nodiscard]] const DNTTTable &gpu_rns_tables() const noexcept { return gpu_rns_tables_; }

    DNTTTable &gpu_rns_tables() { return gpu_rns_tables_; }

    [[nodiscard]] auto *coeff_div_plain() const { return coeff_div_plain_.get(); }

    [[nodiscard]] auto *coeff_div_plain_shoup() const { return coeff_div_plain_shoup_.get(); }

    [[nodiscard]] auto *plain_modulus() const { return plain_modulus_.get(); }

    [[nodiscard]] auto *plain_modulus_shoup() const { return plain_modulus_shoup_.get(); }

    [[nodiscard]] auto *plain_upper_half_increment() const { return plain_upper_half_increment_.get(); }

};
