// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/kswitchkeys.h"
#include <stdexcept>

using namespace std;
using namespace seal::util;

namespace seal
{
    KSwitchKeys &KSwitchKeys::operator=(const KSwitchKeys &assign)
    {
         
        if (this == &assign)
        {
            return *this;
        }

        parms_id_ = assign.parms_id_;

        keys_.clear();
        size_t keys_dim1 = assign.keys_.size();
        keys_.reserve(keys_dim1);
        for (size_t i = 0; i < keys_dim1; i++)
        {
            size_t keys_dim2 = assign.keys_[i].size();
            keys_.emplace_back();
            keys_[i].reserve(keys_dim2);
            for (size_t j = 0; j < keys_dim2; j++)
            {
                keys_[i].emplace_back(PublicKey(pool_));
                keys_[i][j] = assign.keys_[i][j];
            }
        }

        return *this;
    }

    void KSwitchKeys::save_members(ostream &stream) const
    {
        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            uint64_t keys_dim1 = static_cast<uint64_t>(keys_.size());

            stream.write(reinterpret_cast<const char *>(&parms_id_), sizeof(parms_id_type));

            stream.write(reinterpret_cast<const char *>(&keys_dim1), sizeof(uint64_t));

            for (size_t index = 0; index < keys_dim1; index++)
            {
                 
                uint64_t keys_dim2 = static_cast<uint64_t>(keys_[index].size());
                stream.write(reinterpret_cast<const char *>(&keys_dim2), sizeof(uint64_t));

                for (size_t j = 0; j < keys_dim2; j++)
                {
                     
                    keys_[index][j].save(stream, compr_mode_type::none);
                }
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
    }

    void KSwitchKeys::load_members(const SEALContext &context, istream &stream, SEAL_MAYBE_UNUSED SEALVersion version)
    {
         
        if (!context.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        vector<vector<PublicKey>> new_keys;

        auto old_except_mask = stream.exceptions();
        try
        {
             
            stream.exceptions(ios_base::badbit | ios_base::failbit);

            stream.read(reinterpret_cast<char *>(&parms_id_), sizeof(parms_id_type));

            uint64_t keys_dim1 = 0;
            stream.read(reinterpret_cast<char *>(&keys_dim1), sizeof(uint64_t));

            new_keys.reserve(safe_cast<size_t>(keys_dim1));

            for (size_t index = 0; index < keys_dim1; index++)
            {
                 
                uint64_t keys_dim2 = 0;
                stream.read(reinterpret_cast<char *>(&keys_dim2), sizeof(uint64_t));

                new_keys.emplace_back();
                new_keys.back().reserve(safe_cast<size_t>(keys_dim2));
                for (size_t j = 0; j < keys_dim2; j++)
                {
                    PublicKey key(pool_);
                    key.unsafe_load(context, stream);
                    new_keys[index].emplace_back(move(key));
                }
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

        swap(keys_, new_keys);
    }
}
