#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "phantom.h"
#include "source/ckks_evaluator_parallel.cuh"
#include "source/fhe_transformer/fhe_types.cuh"
#include "source/fhe_transformer/fhe_types.cuh"

namespace fhe_transformer {

struct FHELevelMap {
    int max_logical_level = 12;
    size_t top_chain_index = 1;

    static FHELevelMap from_context(const PhantomContext &context, int max_logical_level = 12)
    {
        FHELevelMap map;
        map.max_logical_level = max_logical_level;
        map.top_chain_index = context.get_first_index();
        map.validate_context(context);
        return map;
    }

    static FHELevelMap from_context_top_native_depth(const PhantomContext &context,
                                                     int max_logical_level,
                                                     size_t top_native_chain_depth)
    {
        FHELevelMap map;
        map.max_logical_level = max_logical_level;
        bool found = false;
        for (size_t chain_index = 0; chain_index < context.total_parm_size(); ++chain_index)
        {
            if (context.get_context_data(chain_index).chain_depth() == top_native_chain_depth)
            {
                map.top_chain_index = chain_index;
                found = true;
                break;
            }
        }
        if (!found)
        {
            std::ostringstream oss;
            oss << "FHELevelMap cannot find native chain_depth=" << top_native_chain_depth
                << " in context total_parm_size=" << context.total_parm_size();
            throw std::invalid_argument(oss.str());
        }
        map.validate_context(context);
        return map;
    }

    void validate_context(const PhantomContext &context) const
    {
        const size_t bottom_chain = chain_index_for_logical_level_unchecked(0);
        if (top_chain_index >= context.total_parm_size() || bottom_chain >= context.total_parm_size())
        {
            std::ostringstream oss;
            oss << "FHELevelMap cannot map logical 0.." << max_logical_level
                << " onto Phantom chain indices. top_chain_index=" << top_chain_index
                << " bottom_chain=" << bottom_chain
                << " total_parm_size=" << context.total_parm_size();
            throw std::invalid_argument(oss.str());
        }
    }

    size_t chain_index_for_logical_level_unchecked(int logical_level) const
    {
        return top_chain_index + static_cast<size_t>(max_logical_level - logical_level);
    }

    size_t chain_index_for_logical_level(const PhantomContext &context, int logical_level) const
    {
        if (logical_level < 0 || logical_level > max_logical_level)
        {
            std::ostringstream oss;
            oss << "logical level out of range: " << logical_level
                << ", expected 0.." << max_logical_level;
            throw std::invalid_argument(oss.str());
        }
        const size_t chain_index = chain_index_for_logical_level_unchecked(logical_level);
        if (chain_index >= context.total_parm_size())
        {
            std::ostringstream oss;
            oss << "logical level " << logical_level
                << " maps to invalid Phantom chain_index=" << chain_index
                << ", total_parm_size=" << context.total_parm_size();
            throw std::invalid_argument(oss.str());
        }
        return chain_index;
    }
};

inline int get_level(const FHELevelMap &map, const PhantomCiphertext &ct)
{
    if (ct.chain_index() < map.top_chain_index)
    {
        std::ostringstream oss;
        oss << "ciphertext chain_index=" << ct.chain_index()
            << " is above configured top_chain_index=" << map.top_chain_index;
        throw std::invalid_argument(oss.str());
    }
    const size_t dropped = ct.chain_index() - map.top_chain_index;
    return map.max_logical_level - static_cast<int>(dropped);
}

inline int get_level(const FHEBlockTensor &tensor)
{
    return tensor.level;
}

inline void refresh_tensor_metadata(const PhantomContext &context,
                                    const FHELevelMap &map,
                                    FHEBlockTensor &tensor)
{
    if (tensor.cts.empty())
    {
        tensor.level = -1;
        tensor.scale = 0.0;
        return;
    }
    const int first_level = get_level(map, tensor.cts.front());
    const double first_scale = tensor.cts.front().scale();
    for (size_t i = 0; i < tensor.cts.size(); ++i)
    {
        if (!std::isfinite(tensor.cts[i].scale()) || tensor.cts[i].scale() <= 0)
            throw std::invalid_argument("Invalid CKKS scale in tensor " + tensor.name);
        const int level = get_level(map, tensor.cts[i]);
        if (level != first_level)
        {
            std::ostringstream oss;
            oss << "tensor " << tensor.name << " has mixed levels: cts[0]="
                << first_level << ", cts[" << i << "]=" << level;
            throw std::invalid_argument(oss.str());
        }
    }
    tensor.level = first_level;
    tensor.scale = first_scale;
    (void)context;
}

inline void assert_level(const FHELevelMap &map,
                         const PhantomCiphertext &ct,
                         int expected_level,
                         const std::string &name)
{
    const int actual = get_level(map, ct);
    if (actual != expected_level)
    {
        std::ostringstream oss;
        oss << name << " level mismatch: actual=" << actual
            << " expected=" << expected_level
            << " chain_index=" << ct.chain_index()
            << ". Low-level ciphertexts cannot be promoted without explicit BTS/FB.";
        
        throw std::runtime_error(oss.str());
    }
}

inline void assert_level(const FHEBlockTensor &tensor, int expected_level, const std::string &name)
{
    if (tensor.level != expected_level)
    {
        std::ostringstream oss;
        oss << name << " level mismatch: actual=" << tensor.level
            << " expected=" << expected_level
            << ". Low-level tensors cannot be promoted without explicit BTS/FB.";
        
        throw std::runtime_error(oss.str());
    }
}

inline void align_down_to(const PhantomContext &context, const FHELevelMap &map, PhantomCiphertext &ct, int target_level, const std::string &name, const phantom::util::cuda_stream_wrapper &stream_wrapper =
                              *phantom::util::global_variables::default_stream)
{
    const int current_level = get_level(map, ct);
    if (current_level == target_level)
    {
        
        return;
    }
    if (current_level < target_level)
    {
        std::ostringstream oss;
        oss << name << " cannot align " << current_level << "->" << target_level
            << ". Low-level ciphertext cannot be promoted except by explicit ordinary BTS or FB.";
        
        throw std::runtime_error(oss.str());
    }

    const size_t target_chain_index = map.chain_index_for_logical_level(context, target_level);
    
    ::mod_switch_to_inplace(context, ct, target_chain_index, stream_wrapper);
    
}

inline void align_down_to(const PhantomContext &context, const FHELevelMap &map, FHEBlockTensor &tensor, int target_level, const std::string &name, const phantom::util::cuda_stream_wrapper &stream_wrapper =
                              *phantom::util::global_variables::default_stream)
{
    for (size_t i = 0; i < tensor.cts.size(); ++i)
    {
        align_down_to(context, map, tensor.cts[i], target_level, name + "/ct_" + std::to_string(i), stream_wrapper);
    }
    refresh_tensor_metadata(context, map, tensor);
}

inline void assert_same_level_for_add(const FHELevelMap &map,
                                      const PhantomCiphertext &a,
                                      const PhantomCiphertext &b,
                                      const std::string &name_a,
                                      const std::string &name_b)
{
    const int level_a = get_level(map, a);
    const int level_b = get_level(map, b);
    if (level_a != level_b)
    {
        std::ostringstream oss;
        oss << "add requires same level: " << name_a << "=" << level_a
            << ", " << name_b << "=" << level_b;
        
        throw std::runtime_error(oss.str());
    }
}

inline void assert_same_level_for_mul(const FHELevelMap &map,
                                      const PhantomCiphertext &a,
                                      const PhantomCiphertext &b,
                                      const std::string &name_a,
                                      const std::string &name_b)
{
    const int level_a = get_level(map, a);
    const int level_b = get_level(map, b);
    if (level_a != level_b)
    {
        std::ostringstream oss;
        oss << "mul requires same level: " << name_a << "=" << level_a
            << ", " << name_b << "=" << level_b;
        
        throw std::runtime_error(oss.str());
    }
}

inline int align_pair_for_add(const PhantomContext &context, const FHELevelMap &map, PhantomCiphertext &a, PhantomCiphertext &b, const std::string &name_a, const std::string &name_b, const phantom::util::cuda_stream_wrapper &stream_wrapper =
                                  *phantom::util::global_variables::default_stream)
{
    const int target = std::min(get_level(map, a), get_level(map, b));
    align_down_to(context, map, a, target, name_a, stream_wrapper);
    align_down_to(context, map, b, target, name_b, stream_wrapper);
    assert_same_level_for_add(map, a, b, name_a, name_b);
    return target;
}

inline int align_pair_for_mul(const PhantomContext &context, const FHELevelMap &map, PhantomCiphertext &a, PhantomCiphertext &b, const std::string &name_a, const std::string &name_b, const phantom::util::cuda_stream_wrapper &stream_wrapper =
                                  *phantom::util::global_variables::default_stream)
{

    const int target = std::min(get_level(map, a), get_level(map, b));
    align_down_to(context, map, a, target, name_a, stream_wrapper);
    align_down_to(context, map, b, target, name_b, stream_wrapper);
    assert_same_level_for_mul(map, a, b, name_a, name_b);

    return target;
}

inline void assert_scale_close(const PhantomCiphertext &a,
                               const PhantomCiphertext &b,
                               const std::string &name_a,
                               const std::string &name_b,
                               double relative_tolerance = 1e-3)
{
    const double denom = std::max({std::abs(a.scale()), std::abs(b.scale()), 1.0});
    const double rel = std::abs(a.scale() - b.scale()) / denom;
    if (!std::isfinite(a.scale()) || !std::isfinite(b.scale()) ||
        a.scale() <= 0 || b.scale() <= 0 || !std::isfinite(rel) || rel > relative_tolerance)
    {
        std::ostringstream oss;
        oss << "scale mismatch: " << name_a << "=" << a.scale()
            << ", " << name_b << "=" << b.scale()
            << ", relative=" << rel;
        
        throw std::runtime_error(oss.str());
    }
}

}
