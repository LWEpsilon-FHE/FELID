#pragma once

#include "defines.h"
#include "hestdparms.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <limits>

namespace phantom::arith {
     
    class Modulus {
    public:
         
        Modulus(std::uint64_t value = 0) {
            set_value(value);
        }

        Modulus(const Modulus &copy) = default;

        Modulus(Modulus &&source) = default;

        Modulus &operator=(const Modulus &assign) = default;

        Modulus &operator=(Modulus &&assign) = default;

        inline Modulus &operator=(std::uint64_t value) {
            set_value(value);
            return *this;
        }

        [[nodiscard]] inline int bit_count() const noexcept {
            return bit_count_;
        }

        [[nodiscard]] inline std::size_t uint64_count() const noexcept {
            return uint64_count_;
        }

        [[nodiscard]] inline const uint64_t *data() const noexcept {
            return &value_;
        }

        [[nodiscard]] inline std::uint64_t value() const noexcept {
            return value_;
        }

        [[nodiscard]] inline auto &const_ratio() const noexcept {
            return const_ratio_;
        }

        [[nodiscard]] inline bool is_zero() const noexcept {
            return value_ == 0;
        }

        [[nodiscard]] inline bool is_prime() const noexcept {
            return is_prime_;
        }

        [[nodiscard]] inline bool operator==(const Modulus &compare) const noexcept {
            return value_ == compare.value_;
        }

        [[nodiscard]] inline bool operator==(std::uint64_t compare) const noexcept {
            return value_ == compare;
        }

        [[nodiscard]] inline bool operator!=(const Modulus &compare) const noexcept {
            return !operator==(compare);
        }

        [[nodiscard]] inline bool operator!=(std::uint64_t compare) const noexcept {
            return !operator==(compare);
        }

        [[nodiscard]] inline bool operator<(const Modulus &compare) const noexcept {
            return value_ < compare.value_;
        }

        [[nodiscard]] inline bool operator<(std::uint64_t compare) const noexcept {
            return value_ < compare;
        }

        [[nodiscard]] inline bool operator<=(const Modulus &compare) const noexcept {
            return value_ <= compare.value_;
        }

        [[nodiscard]] inline bool operator<=(std::uint64_t compare) const noexcept {
            return value_ <= compare;
        }

        [[nodiscard]] inline bool operator>(const Modulus &compare) const noexcept {
            return value_ > compare.value_;
        }

        [[nodiscard]] inline bool operator>(std::uint64_t compare) const noexcept {
            return value_ > compare;
        }

        [[nodiscard]] inline bool operator>=(const Modulus &compare) const noexcept {
            return value_ >= compare.value_;
        }

        [[nodiscard]] inline bool operator>=(std::uint64_t compare) const noexcept {
            return value_ >= compare;
        }

        [[nodiscard]] std::uint64_t reduce(std::uint64_t value) const;

        struct ModulusPrivateHelper;

        void save(std::ostream &stream) const {
            save_members(stream);
        }

        void load(std::istream &stream) {
            load_members(stream);
        }

    private:
        void set_value(std::uint64_t value);

        void save_members(std::ostream &stream) const;

        void load_members(std::istream &stream);

        std::uint64_t value_ = 0;

        std::array<std::uint64_t, 3> const_ratio_{{0, 0, 0}};

        std::size_t uint64_count_ = 0;

        int bit_count_ = 0;

        bool is_prime_ = false;
    };

    enum class sec_level_type : int {
         
        none = 0,

        tc128 = 128,

        tc192 = 192,

        tc256 = 256
    };

    class CoeffModulus {
    public:
        CoeffModulus() = delete;

        [[nodiscard]] static constexpr int MaxBitCount(
                std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128) noexcept {
            switch (sec_level) {
                case sec_level_type::tc128:
                    return util::he_std_parms_128_tc(poly_modulus_degree);

                case sec_level_type::tc192:
                    return util::he_std_parms_192_tc(poly_modulus_degree);

                case sec_level_type::tc256:
                    return util::he_std_parms_256_tc(poly_modulus_degree);

                case sec_level_type::none:
                    return (std::numeric_limits<int>::max)();

                default:
                    return 0;
            }
        }

        [[nodiscard]] static std::vector<Modulus> BFVDefault(
                std::size_t poly_modulus_degree, sec_level_type sec_level = sec_level_type::tc128);

        [[nodiscard]] static std::vector<Modulus>
        Create(std::size_t poly_modulus_degree, const std::vector<int> &bit_sizes);
    };

    class PlainModulus {
    public:
        PlainModulus() = delete;

        [[nodiscard]] static inline Modulus Batching(std::size_t poly_modulus_degree, int bit_size) {
            return CoeffModulus::Create(poly_modulus_degree, {bit_size})[0];
        }
    };
}
