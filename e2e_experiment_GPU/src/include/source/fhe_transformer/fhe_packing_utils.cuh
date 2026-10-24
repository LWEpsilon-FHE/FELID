#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_level_utils.cuh"
#include "source/fhe_transformer/fhe_types.cuh"

namespace fhe_transformer {

inline FHEBlockTensor make_metadata_view(const FHEBlockTensor &input,
                                         std::vector<PhantomCiphertext> cts,
                                         const std::string &name,
                                         std::vector<int> shape,
                                         PackingType packing)
{
    FHEBlockTensor out;
    out.cts = std::move(cts);
    out.name = name;
    out.logical_shape = std::move(shape);
    out.packing = packing;
    out.level = input.level;
    out.scale = input.scale;
    out.num_heads = input.num_heads;
    out.head_dim = input.head_dim;
    out.is_complex = input.is_complex;
    return out;
}

inline std::vector<FHEBlockTensor> split_heads(const FHEBlockTensor &input,
                                               int num_heads = 12,
                                               int head_dim = 64)
{
    if (input.logical_shape.size() != 2)
    {
        throw std::runtime_error("split_heads requires rank-2 logical shape");
    }
    if (input.logical_shape[1] != num_heads * head_dim)
    {
        throw std::runtime_error("split_heads shape mismatch: hidden dim must equal num_heads * head_dim");
    }
    if (num_heads <= 0 || input.cts.size() % static_cast<size_t>(num_heads) != 0)
    {
        throw std::runtime_error("split_heads requires ciphertext count divisible by num_heads");
    }

    const size_t cts_per_head = input.cts.size() / static_cast<size_t>(num_heads);
    std::vector<FHEBlockTensor> heads;
    heads.reserve(num_heads);
    for (int h = 0; h < num_heads; ++h)
    {
        const size_t begin = static_cast<size_t>(h) * cts_per_head;
        const size_t end = begin + cts_per_head;
        std::vector<PhantomCiphertext> cts(input.cts.begin() + begin, input.cts.begin() + end);
        FHEBlockTensor head = make_metadata_view(input, std::move(cts),
                                                 input.name + "_head_" + (h < 10 ? "0" : "") + std::to_string(h),
                                                 {input.logical_shape[0], head_dim},
                                                 input.packing);
        head.num_heads = num_heads;
        head.head_dim = head_dim;
        heads.push_back(std::move(head));
        
    }
    return heads;
}

inline FHEBlockTensor merge_heads(const PhantomContext &context, const FHELevelMap &map, std::vector<FHEBlockTensor> heads, const std::string &out_name)
{
    if (heads.empty())
    {
        throw std::runtime_error("merge_heads requires at least one head");
    }
    const int num_heads = static_cast<int>(heads.size());
    const int seq_len = heads.front().logical_shape.at(0);
    const int head_dim = heads.front().logical_shape.at(1);
    int min_level = heads.front().level;
    for (const auto &head : heads)
    {
        if (head.logical_shape.size() != 2 || head.logical_shape[0] != seq_len || head.logical_shape[1] != head_dim)
        {
            throw std::runtime_error("merge_heads requires all heads to have identical rank-2 head shapes");
        }
        min_level = std::min(min_level, head.level);
    }

    for (auto &head : heads)
    {
        align_down_to(context, map, head, min_level, head.name);
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = {seq_len, num_heads * head_dim};
    out.packing = heads.front().packing;
    out.level = min_level;
    out.scale = heads.front().scale;
    out.num_heads = num_heads;
    out.head_dim = head_dim;
    out.is_complex = heads.front().is_complex;
    for (auto &head : heads)
    {
        out.cts.insert(out.cts.end(),
                       std::make_move_iterator(head.cts.begin()),
                       std::make_move_iterator(head.cts.end()));
    }
    refresh_tensor_metadata(context, map, out);
    
    return out;
}

inline FHEBlockTensor metadata_packing_convert(const FHEBlockTensor &input,
                                               const std::string &out_name,
                                               PackingType output_packing,
                                               const std::string &helper_name)
{
    
    FHEBlockTensor out = input;
    out.name = out_name;
    out.packing = output_packing;
    
    return out;
}

inline FHEBlockTensor ext_QK(const FHEBlockTensor &input, const std::string &out_name)
{
    return metadata_packing_convert(input, out_name, PackingType::InterleavedDiagonal, "ext_QK");
}

inline FHEBlockTensor rot_K(const FHEBlockTensor &input, const std::string &out_name)
{
    return metadata_packing_convert(input, out_name, input.packing, "rot_K");
}

inline FHEBlockTensor real_Q(const FHEBlockTensor &input, const std::string &out_name)
{
    return metadata_packing_convert(input, out_name, PackingType::InterleavedColumn, "real_Q");
}

}
