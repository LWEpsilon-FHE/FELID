// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/plaintext.h"
#include "seal/util/common.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        bool is_dec_char(char c)
        {
            return c >= '0' && c <= '9';
        }

        int get_dec_value(char c)
        {
            return c - '0';
        }

        int get_coeff_length(const char *poly)
        {
            int length = 0;
            while (is_hex_char(*poly))
            {
                length++;
                poly++;
            }
            return length;
        }

        int get_coeff_power(const char *poly, int *power_length)
        {
            int length = 0;
            if (*poly == '\0')
            {
                *power_length = 0;
                return 0;
            }
            if (*poly != 'x')
            {
                return -1;
            }
            poly++;
            length++;

            if (*poly != '^')
            {
                return -1;
            }
            poly++;
            length++;

            int power = 0;
            while (is_dec_char(*poly))
            {
                power *= 10;
                power += get_dec_value(*poly);
                poly++;
                length++;
            }
            *power_length = length;
            return power;
        }

        int get_plus(const char *poly)
        {
            if (*poly == '\0')
            {
                return 0;
            }
            if (*poly++ != ' ')
            {
                return -1;
            }
            if (*poly++ != '+')
            {
                return -1;
            }
            if (*poly != ' ')
            {
                return -1;
            }
            return 3;
        }
    }  

    Plaintext &Plaintext::operator=(const string &hex_poly)
    {
        if (is_ntt_form())
        {
            throw logic_error("cannot set an NTT transformed Plaintext");
        }
        if (unsigned_gt(hex_poly.size(), numeric_limits<int>::max()))
        {
            throw invalid_argument("hex_poly too long");
        }
        int length = safe_cast<int>(hex_poly.size());

        int assign_coeff_count = 0;

        int assign_coeff_bit_count = 0;
        int pos = 0;
        int last_power = safe_cast<int>(min(data_.max_size(), safe_cast<size_t>(numeric_limits<int>::max())));
        const char *hex_poly_ptr = hex_poly.data();
        while (pos < length)
        {
             
            int coeff_length = get_coeff_length(hex_poly_ptr + pos);
            if (coeff_length == 0)
            {
                throw invalid_argument("unable to parse hex_poly");
            }

            int coeff_bit_count = get_hex_string_bit_count(hex_poly_ptr + pos, coeff_length);
            if (coeff_bit_count > assign_coeff_bit_count)
            {
                assign_coeff_bit_count = coeff_bit_count;
            }
            pos += coeff_length;

            int power_length = 0;
            int power = get_coeff_power(hex_poly_ptr + pos, &power_length);
            if (power == -1 || power >= last_power)
            {
                throw invalid_argument("unable to parse hex_poly");
            }
            if (assign_coeff_count == 0)
            {
                assign_coeff_count = power + 1;
            }
            pos += power_length;
            last_power = power;

            int plus_length = get_plus(hex_poly_ptr + pos);
            if (plus_length == -1)
            {
                throw invalid_argument("unable to parse hex_poly");
            }
            pos += plus_length;
        }

        if (assign_coeff_count == 0 || assign_coeff_bit_count == 0)
        {
            set_zero();
            return *this;
        }

        if (assign_coeff_bit_count > bits_per_uint64)
        {
            throw invalid_argument("hex_poly has too large coefficients");
        }
        resize(safe_cast<size_t>(assign_coeff_count));

        pos = 0;
        last_power = safe_cast<int>(coeff_count());
        while (pos < length)
        {
             
            const char *coeff_start = hex_poly_ptr + pos;
            int coeff_length = get_coeff_length(coeff_start);
            pos += coeff_length;

            int power_length = 0;
            int power = get_coeff_power(hex_poly_ptr + pos, &power_length);
            pos += power_length;

            int plus_length = get_plus(hex_poly_ptr + pos);
            pos += plus_length;

            for (int zero_power = last_power - 1; zero_power > power; --zero_power)
            {
                data_[static_cast<size_t>(zero_power)] = 0;
            }

            uint64_t *coeff_ptr = data_.begin() + power;
            hex_string_to_uint(coeff_start, coeff_length, size_t(1), coeff_ptr);
            last_power = power;
        }

        for (int zero_power = last_power - 1; zero_power >= 0; --zero_power)
        {
            data_[static_cast<size_t>(zero_power)] = 0;
        }

        return *this;
    }

    void Plaintext::save_members(ostream &stream) const
    {
        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(parms_id_type));
            uint64_t coeff_count64 = static_cast<uint64_t>(coeff_count_);
            stream.write(reinterpret_cast<const char *>(&coeff_count64), sizeof(uint64_t));
            stream.write(reinterpret_cast<const char *>(&scale_), sizeof(double));
            data_.save(stream, compr_mode_type::none);
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            throw runtime_error("I/O error");
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);
    }

    void Plaintext::load_members(const SEALContext &context, istream &stream, SEAL_MAYBE_UNUSED SEALVersion version)
    {
         
        if (!context.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        Plaintext new_data(data_.pool());

        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            parms_id_type parms_id{};
            stream.read(reinterpret_cast<char *>(&parms_id), sizeof(parms_id_type));

            uint64_t coeff_count64 = 0;
            stream.read(reinterpret_cast<char *>(&coeff_count64), sizeof(uint64_t));

            double scale = 0;
            stream.read(reinterpret_cast<char *>(&scale), sizeof(double));

            new_data.parms_id_ = parms_id;
            new_data.coeff_count_ = safe_cast<size_t>(coeff_count64);
            new_data.scale_ = scale;

            if (!is_metadata_valid_for(new_data, context, true))
            {
                throw logic_error("plaintext data is invalid");
            }

            new_data.data_.reserve(new_data.coeff_count_);

            new_data.data_.load(stream, new_data.coeff_count_);

            if (!is_buffer_valid(new_data))
            {
                throw logic_error("plaintext data is invalid");
            }
        }
        catch (const ios_base::failure &)
        {
            stream.exceptions(old_except_mask);
            throw runtime_error("I/O error");
        }
        catch (...)
        {
            stream.exceptions(old_except_mask);
            throw;
        }
        stream.exceptions(old_except_mask);

        swap(*this, new_data);
    }
}
