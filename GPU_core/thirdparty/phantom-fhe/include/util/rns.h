#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "encryptionparams.h"
#include "modulus.h"
#include "ntt.h"
#include "uintarithsmallmod.h"

namespace phantom::arith {
    class RNSBase {

    private:
        bool initialize();

        std::size_t size_;
         
        std::vector<Modulus> mod_;
         
        std::vector<std::uint64_t> prod_mod_;
         
        std::vector<std::uint64_t> prod_hat_;
         
        std::vector<uint64_t> hat_mod_;
        std::vector<uint64_t> hat_mod_shoup_;
         
        std::vector<uint64_t> hatInv_mod_;
        std::vector<uint64_t> hatInv_mod_shoup_;
         
        std::vector<double> inv_;

    public:
        RNSBase() : size_(0) {}

        explicit RNSBase(const std::vector<Modulus> &rnsbase);

        void init(const std::vector<Modulus> &rnsbase);

        RNSBase(const RNSBase &copy);

        void init(const RNSBase &copy);

        RNSBase(RNSBase &&source) = default;

        RNSBase &operator=(const RNSBase &assign) = delete;

        [[nodiscard]] inline const Modulus &operator[](std::size_t index) const {
            if (index >= size_) {
                throw std::out_of_range("index is out of range");
            }
            return mod_[index];
        }

        [[nodiscard]] std::size_t size() const noexcept { return size_; }

        [[nodiscard]] bool contains(const Modulus &value) const noexcept;

        [[nodiscard]] bool is_subbase_of(const RNSBase &superbase) const noexcept;

        [[nodiscard]] inline bool is_superbase_of(const RNSBase &subbase) const noexcept {
            return subbase.is_subbase_of(*this);
        }

        [[maybe_unused]] [[nodiscard]] bool is_proper_subbase_of(const RNSBase &superbase) const noexcept {
            return (size_ < superbase.size_) && is_subbase_of(superbase);
        }

        [[maybe_unused]] [[nodiscard]] bool is_proper_superbase_of(const RNSBase &subbase) const noexcept {
            return (size_ > subbase.size_) && !is_subbase_of(subbase);
        }

        [[nodiscard]] RNSBase extend(const Modulus &value) const;

        [[nodiscard]] RNSBase extend(const RNSBase &other) const;

        [[nodiscard]] RNSBase drop() const;

        [[nodiscard]] RNSBase drop(const Modulus &value) const;

        [[nodiscard]] RNSBase drop(const std::vector<Modulus> &values) const;

        void decompose(std::uint64_t *value) const;

        void decompose_array(std::uint64_t *value, std::size_t count) const;

        void compose(std::uint64_t *value) const;

        void compose_array(std::uint64_t *value, std::size_t count) const;

        [[nodiscard]] const Modulus *base() const noexcept { return mod_.data(); }

        [[nodiscard]] const std::uint64_t *big_modulus() const noexcept { return prod_mod_.data(); }

        [[nodiscard]] const std::uint64_t *big_qiHat() const noexcept { return prod_hat_.data(); }

        [[nodiscard]] const uint64_t *qiHat_mod_qi() const noexcept { return hat_mod_.data(); }

        [[nodiscard]] const uint64_t *qiHat_mod_qi_shoup() const noexcept { return hat_mod_shoup_.data(); }

        [[nodiscard]] const uint64_t *QHatInvModq() const noexcept { return hatInv_mod_.data(); }

        [[nodiscard]] const uint64_t *QHatInvModq_shoup() const noexcept { return hatInv_mod_shoup_.data(); }

        [[nodiscard]] const double *inv() const noexcept { return inv_.data(); }

    };

    class BaseConverter {

    private:
        void initialize();

        RNSBase ibase_;
        RNSBase obase_;
        std::vector<std::vector<std::uint64_t>> QHatModp_;
        std::vector<std::vector<std::uint64_t>> alphaQModp_;
        std::vector<uint64_t> negPQHatInvModq_;
        std::vector<uint64_t> negPQHatInvModq_shoup_;
        std::vector<std::vector<std::uint64_t>> QInvModp_;
        std::vector<uint64_t> PModq_;
        std::vector<uint64_t> PModq_shoup_;

    public:
        BaseConverter(const RNSBase &ibase, const RNSBase &obase) : ibase_(std::move(ibase)), obase_(std::move(obase)) {
            initialize();
        }

        BaseConverter(BaseConverter &&source) = delete;

        BaseConverter &operator=(const BaseConverter &assign) = delete;

        BaseConverter &operator=(BaseConverter &&assign) = delete;

        size_t ibase_size() const noexcept { return ibase_.size(); }

        size_t obase_size() const noexcept { return obase_.size(); }

        const RNSBase &ibase() const noexcept { return ibase_; }

        const RNSBase &obase() const noexcept { return obase_; }

        uint64_t *QHatModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QHatModp index is out of range");

            return QHatModp_[index].data();
        }

        uint64_t *alphaQModp(size_t index) {
            if (index >= ibase_size() + 1)
                throw std::out_of_range("alphaQModp index is out of range");

            return alphaQModp_[index].data();
        }

        auto *negPQHatInvModq() { return negPQHatInvModq_.data(); }

        auto *negPQHatInvModq_shoup() { return negPQHatInvModq_shoup_.data(); }

        uint64_t *QInvModp(size_t index) {
            if (index >= obase_size())
                throw std::out_of_range("QInvModp index is out of range");

            return QInvModp_[index].data();
        }

        auto *PModq() { return PModq_.data(); }

        auto *PModq_shoup() { return PModq_shoup_.data(); }

    };
}
