// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/numth.h"
#include "util/uintarithsmallmod.h"
#include "util/uintcore.h"
#include <random>

using namespace std;

namespace phantom {
    namespace arith {
        vector <uint64_t> conjugate_classes(uint64_t modulus, uint64_t subgroup_generator) {
            vector<uint64_t> classes{};
            for (uint64_t i = 0; i < modulus; i++) {
                if (gcd(i, modulus) > 1) {
                    classes.push_back(0);
                } else {
                    classes.push_back(i);
                }
            }
            for (uint64_t i = 0; i < modulus; i++) {
                if (classes[static_cast<size_t>(i)] == 0) {
                    continue;
                }
                if (classes[static_cast<size_t>(i)] < i) {
                     
                    classes[static_cast<size_t>(i)] = classes[static_cast<size_t>(classes[static_cast<size_t>(i)])];
                    continue;
                }
                 
                uint64_t j = (i * subgroup_generator) % modulus;
                while (classes[static_cast<size_t>(j)] != i) {

                    classes[static_cast<size_t>(classes[static_cast<size_t>(j)])] = i;
                    j = (j * subgroup_generator) % modulus;
                }
            }
            return classes;
        }

        bool try_invert_uint_mod(uint64_t value, uint64_t modulus, uint64_t &result) {
            if (value == 0) {
                return false;
            }
            auto gcd_tuple = xgcd(value, modulus);
            if (get<0>(gcd_tuple) != 1) {
                return false;
            } else if (get<1>(gcd_tuple) < 0) {
                result = static_cast<uint64_t>(get<1>(gcd_tuple)) + modulus;
                return true;
            } else {
                result = static_cast<uint64_t>(get<1>(gcd_tuple));
                return true;
            }
        }

        vector <uint64_t> multiplicative_orders(vector <uint64_t> conjugate_classes, uint64_t modulus) {
            vector<uint64_t> orders{};
            orders.push_back(0);
            orders.push_back(1);

            for (uint64_t i = 2; i < modulus; i++) {
                if (conjugate_classes[static_cast<size_t>(i)] <= 1) {
                    orders.push_back(conjugate_classes[static_cast<size_t>(i)]);
                    continue;
                }
                if (conjugate_classes[static_cast<size_t>(i)] < i) {
                    orders.push_back(orders[static_cast<size_t>(conjugate_classes[static_cast<size_t>(i)])]);
                    continue;
                }
                uint64_t j = (i * i) % modulus;
                uint64_t order = 2;
                while (conjugate_classes[static_cast<size_t>(j)] != 1) {
                    j = (j * i) % modulus;
                    order++;
                }
                orders.push_back(order);
            }
            return orders;
        }

        void babystep_giantstep(uint64_t modulus, vector <uint64_t> &baby_steps, vector <uint64_t> &giant_steps) {
            int exponent = get_power_of_two(modulus);
            if (exponent < 0) {
                throw invalid_argument("modulus must be a power of 2");
            }

            uint64_t k = uint64_t(1) << (exponent / 2);
            uint64_t l = modulus / k;

            baby_steps.clear();
            giant_steps.clear();

            uint64_t m = mul_safe(modulus, uint64_t(2));
            uint64_t g = 3;  
            uint64_t kprime = k >> 1;
            uint64_t value = 1;
            for (uint64_t i = 0; i < kprime; i++) {
                baby_steps.push_back(value);
                baby_steps.push_back(m - value);
                value = mul_safe(value, g) % m;
            }

            uint64_t value2 = value;
            for (uint64_t j = 0; j < l; j++) {
                giant_steps.push_back(value2);
                value2 = mul_safe(value2, value) % m;
            }
        }

        pair <size_t, size_t> decompose_babystep_giantstep(
                uint64_t modulus, uint64_t input, const vector <uint64_t> &baby_steps,
                const vector <uint64_t> &giant_steps) {
            for (size_t i = 0; i < giant_steps.size(); i++) {
                uint64_t gs = giant_steps[i];
                for (size_t j = 0; j < baby_steps.size(); j++) {
                    uint64_t bs = baby_steps[j];
                    if (mul_safe(gs, bs) % modulus == input) {
                        return {i, j};
                    }
                }
            }
            throw logic_error("failed to decompose input");
        }

        bool is_prime(const Modulus &modulus, size_t num_rounds) {
            uint64_t value = modulus.value();
             
            if (value < 2) {
                return false;
            }
            if (2 == value) {
                return true;
            }
            if (0 == (value & 0x1)) {
                return false;
            }
            if (3 == value) {
                return true;
            }
            if (0 == (value % 3)) {
                return false;
            }
            if (5 == value) {
                return true;
            }
            if (0 == (value % 5)) {
                return false;
            }
            if (7 == value) {
                return true;
            }
            if (0 == (value % 7)) {
                return false;
            }
            if (11 == value) {
                return true;
            }
            if (0 == (value % 11)) {
                return false;
            }
            if (13 == value) {
                return true;
            }
            if (0 == (value % 13)) {
                return false;
            }

            uint64_t d = value - 1;
            uint64_t r = 0;
            while (0 == (d & 0x1)) {
                d >>= 1;
                r++;
            }
            if (r == 0) {
                return false;
            }

            random_device rand;
            uniform_int_distribution<unsigned long long> dist(3, value - 1);
            for (size_t i = 0; i < num_rounds; i++) {
                uint64_t a = i ? dist(rand) : 2;
                uint64_t x = exponentiate_uint_mod(a, d, modulus);
                if (x == 1 || x == value - 1) {
                    continue;
                }
                uint64_t count = 0;
                do {
                    x = multiply_uint_mod(x, x, modulus);
                    count++;
                } while (x != value - 1 && count < r - 1);
                if (x != value - 1) {
                    return false;
                }
            }
            return true;
        }

        vector <Modulus> get_primes(size_t ntt_size, int bit_size, size_t count) {
            vector<Modulus> destination;
            uint64_t factor = mul_safe(uint64_t(2), uint64_t(ntt_size));

            uint64_t value = uint64_t(0x1) << bit_size;
            try {
                value = sub_safe(value, factor) + 1;
            }
            catch (const logic_error &) {
                throw logic_error("failed to find enough qualifying primes");
            }

            uint64_t lower_bound = uint64_t(0x1) << (bit_size - 1);
            while (count > 0 && value > lower_bound) {
                Modulus new_mod(value);
                if (new_mod.is_prime()) {
                    destination.emplace_back(move(new_mod));
                    count--;
                }
                value -= factor;
            }
            if (count > 0) {
                throw logic_error("failed to find enough qualifying primes");
            }
            return destination;
        }

        vector <Modulus> get_primes_below(size_t ntt_size, uint64_t upper_bound, size_t count) {
            vector<Modulus> destination;
            uint64_t factor = mul_safe(uint64_t(2), uint64_t(ntt_size));

            uint64_t value = upper_bound;
            int bit_size = get_significant_bit_count(upper_bound);
             
            try {
                value = sub_safe(value, factor);
            }
            catch (const logic_error &) {
                throw logic_error("failed to find enough qualifying primes 1");
            }

            uint64_t lower_bound = uint64_t(0x1) << (bit_size - 1);

            while (count > 0 && value > lower_bound) {
                Modulus new_mod(value);
                if (new_mod.is_prime()) {
                    destination.emplace_back(move(new_mod));
                    count--;
                }
                value -= factor;
            }
            if (count > 0) {
                throw logic_error("failed to find enough qualifying primes 2");
            }
            return destination;
        }

        bool is_primitive_root(uint64_t root, uint64_t degree, const Modulus &modulus) {
            if (root == 0) {
                return false;
            }

            return exponentiate_uint_mod(root, degree >> 1, modulus) == (modulus.value() - 1);
        }

        bool try_primitive_root(uint64_t degree, const Modulus &modulus, uint64_t &destination) {
             
            uint64_t size_entire_group = modulus.value() - 1;

            uint64_t size_quotient_group = size_entire_group / degree;

            if (size_entire_group - size_quotient_group * degree != 0) {
                return false;
            }

            random_device rd;

            int attempt_counter = 0;
            int attempt_counter_max = 100;
            do {
                attempt_counter++;

                destination =
                        barrett_reduce_64((static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd()), modulus);

                destination = exponentiate_uint_mod(destination, size_quotient_group, modulus);
            } while (!is_primitive_root(destination, degree, modulus) && (attempt_counter < attempt_counter_max));

            return is_primitive_root(destination, degree, modulus);
        }

        bool try_minimal_primitive_root(uint64_t degree, const Modulus &modulus, uint64_t &destination) {
            uint64_t root;
            if (!try_primitive_root(degree, modulus, root)) {
                return false;
            }
            uint64_t generator_sq = multiply_uint_mod(root, root, modulus);
            uint64_t current_generator = root;

            for (size_t i = 0; i < degree; i++) {

                if (current_generator < root) {
                    root = current_generator;
                }

                current_generator = multiply_uint_mod(current_generator, generator_sq, modulus);
            }

            destination = root;
            return true;
        }
    }
}
