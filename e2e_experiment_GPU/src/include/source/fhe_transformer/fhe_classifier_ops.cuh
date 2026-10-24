#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_matrix_ops.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

struct ClassifierBackwardOpParams {
    int batch_size = 1;
    int row_X = 128;
    int col_W = 768;
    int pre_blocks = 6;
    int column_blocks = 48;
    bool pre_split_output = true;
};

inline void require_single_ciphertext(const FHEBlockTensor &tensor,
                                      const std::string &op_name)
{
    if (tensor.cts.size() != 1)
    {
        std::ostringstream oss;
        oss << op_name << " expects exactly one ciphertext for " << tensor.name
            << ", got " << tensor.cts.size();
        throw std::runtime_error(oss.str());
    }
}

inline FHEBlockTensor phantom_ext_copy_x(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &enc_X,
                                         int batch_size,
                                         int row_X,
                                         const std::string &out_name,
                                         const std::string &step_name = "ext_copy_X")
{
    
    if (batch_size <= 0 || row_X <= 0 || enc_X.cts.empty())
    {
        throw std::runtime_error(step_name + " requires positive dimensions and non-empty input");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    if (n % (row_X * batch_size) != 0)
    {
        throw std::runtime_error(step_name + " row_X*batch_size must divide slot count");
    }
    if (enc_X.level <= 0)
    {
        throw std::runtime_error(step_name + " cannot consume a level from logical level " +
                                 std::to_string(enc_X.level));
    }

    std::vector<double> mask(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n / (row_X * batch_size); ++i)
    {
        for (int j = 0; j < batch_size; ++j)
        {
            mask[static_cast<size_t>(i * row_X * batch_size + j)] = 1.0;
        }
    }
    PhantomPlaintext encoded_mask;
    encode_values(adapter,
                      mask,
                      enc_X.cts.front().params_id(),
                      enc_X.cts.front().scale(),
                      encoded_mask);

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = enc_X.logical_shape;
    out.packing = enc_X.packing;
    out.is_complex = false;
    out.cts.resize(enc_X.cts.size());

    for (size_t i = 0; i < enc_X.cts.size(); ++i)
    {
        out.cts[i] = enc_X.cts[i];
        adapter.require_evaluator().multiply_plain_inplace(out.cts[i], encoded_mask);
        adapter.require_evaluator().rescale_to_next_inplace(out.cts[i]);

        int rot_index = batch_size;
        while (rot_index < row_X * batch_size)
        {
            PhantomCiphertext tmp;
            adapter.require_evaluator().rotate_vector(out.cts[i],
                                                      -rot_index,
                                                      adapter.require_galois_keys(),
                                                      tmp);
            adapter.require_evaluator().add_inplace(out.cts[i], tmp);
            
            rot_index *= 2;
        }
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_ccmm_class_1_from_ext_batch(FHEAdapterContext &adapter,
                                                          const FHEBlockTensor &ext_X,
                                                          const FHEBlockTensor &W,
                                                          const MatrixOpParams &params,
                                                          const std::string &out_name,
                                                          const std::string &step_name = "CCMM-C1-from-ext")
{
    
    validate_matrix_params(params);
    if (ext_X.is_complex || W.is_complex)
    {
        throw std::runtime_error(step_name + " requires real classifier tensors");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = params.row_X * params.col_X * params.batch_size / n;
    if (mid <= 0 || params.row_X * params.col_X * params.batch_size % n != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not produce integral X ciphertext count");
    }
    if (static_cast<int>(ext_X.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " input ciphertext count mismatch");
    }
    if (static_cast<int>(W.cts.size()) % mid != 0)
    {
        throw std::runtime_error(step_name + " W ciphertext count must be a multiple of mid");
    }
    const int ed = static_cast<int>(W.cts.size()) / mid;
    if (ed <= 0 || ed * params.row_X * params.batch_size > n)
    {
        throw std::runtime_error(step_name + " output block count is incompatible with slot layout");
    }

    FHEBlockTensor weight = W;
    align_down_to(adapter.require_context(), adapter.level_map, weight, ext_X.level, weight.name);
    if (ext_X.level <= 0)
    {
        throw std::runtime_error(step_name + " cannot consume multiply level from ext_X level " +
                                 std::to_string(ext_X.level));
    }

    std::vector<PhantomCiphertext> partial(static_cast<size_t>(ed));
    for (int i = 0; i < ed; ++i)
    {
        bool set = false;
        for (int j = 0; j < mid; ++j)
        {
            PhantomCiphertext lhs = ext_X.cts[static_cast<size_t>(j)];
            PhantomCiphertext rhs = weight.cts[static_cast<size_t>(i * mid + j)];
            align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, ext_X.name + "/ct_" + std::to_string(j), weight.name + "/ct_" + std::to_string(i * mid + j));
            PhantomCiphertext tmp;
            adapter.require_evaluator().multiply(lhs, rhs, tmp);
            
            if (!set)
            {
                partial[static_cast<size_t>(i)] = std::move(tmp);
                set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(partial[static_cast<size_t>(i)], tmp);
                
            }
        }
        adapter.require_evaluator().relinearize_inplace(partial[static_cast<size_t>(i)],
                                                        adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(partial[static_cast<size_t>(i)]);

        int rot_index = n / 2;
        while (rot_index >= params.row_X * params.batch_size)
        {
            PhantomCiphertext tmp;
            adapter.require_evaluator().rotate_vector(partial[static_cast<size_t>(i)],
                                                      rot_index,
                                                      adapter.require_galois_keys(),
                                                      tmp);
            adapter.require_evaluator().add_inplace(partial[static_cast<size_t>(i)], tmp);
            
            rot_index /= 2;
        }
    }

    PhantomCiphertext final_output;
    bool final_set = false;
    for (int i = 0; i < ed; ++i)
    {
        std::vector<double> mask(static_cast<size_t>(n), 0.0);
        for (int slot = i * params.row_X * params.batch_size;
             slot < (i + 1) * params.row_X * params.batch_size;
             ++slot)
        {
            mask[static_cast<size_t>(slot)] = 1.0;
        }
        PhantomPlaintext encoded_mask;
        encode_values(adapter,
                          mask,
                          partial[static_cast<size_t>(i)].params_id(),
                          partial[static_cast<size_t>(i)].scale(),
                          encoded_mask);
        adapter.require_evaluator().multiply_plain_inplace(partial[static_cast<size_t>(i)], encoded_mask);
        adapter.require_evaluator().rescale_to_next_inplace(partial[static_cast<size_t>(i)]);
        
        if (!final_set)
        {
            final_output = std::move(partial[static_cast<size_t>(i)]);
            final_set = true;
        }
        else
        {
            adapter.require_evaluator().add_inplace(final_output, partial[static_cast<size_t>(i)]);
            
        }
    }

    PhantomCiphertext tmp;
    adapter.require_evaluator().rotate_vector(final_output,
                                              n / 2,
                                              adapter.require_galois_keys(),
                                              tmp);
    adapter.require_evaluator().add_inplace(final_output, tmp);

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = {1, params.col_W};
    out.packing = PackingType::InterleavedRow;
    out.is_complex = false;
    out.cts.push_back(std::move(final_output));
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_ccmm_class_2_batch(FHEAdapterContext &adapter,
                                                 const FHEBlockTensor &enc_X,
                                                 const FHEBlockTensor &W,
                                                 const MatrixOpParams &params,
                                                 const std::string &out_name,
                                                 const std::string &step_name = "CCMM-C2")
{
    
    validate_matrix_params(params);
    require_single_ciphertext(enc_X, step_name);
    require_single_ciphertext(W, step_name);
    if (enc_X.is_complex || W.is_complex)
    {
        throw std::runtime_error(step_name + " requires real packed classifier tensors");
    }

    PhantomCiphertext lhs = enc_X.cts.front();
    PhantomCiphertext rhs = W.cts.front();
    const int start_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, enc_X.name + "/ct_0", W.name + "/ct_0");
    if (start_level <= 0)
    {
        throw std::runtime_error(step_name + " cannot consume a level from logical level " +
                                 std::to_string(start_level));
    }

    PhantomCiphertext output = lhs;
    adapter.require_evaluator().multiply_inplace(output, rhs);
    adapter.require_evaluator().relinearize_inplace(output, adapter.require_relin_keys());
    adapter.require_evaluator().rescale_to_next_inplace(output);

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    int rot_index = n / 4;
    while (rot_index >= params.batch_size)
    {
        PhantomCiphertext tmp;
        adapter.require_evaluator().rotate_vector(output,
                                                  rot_index,
                                                  adapter.require_galois_keys(),
                                                  tmp);
        adapter.require_evaluator().add_inplace(output, tmp);
        
        rot_index /= 2;
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = {params.row_X, params.col_W};
    out.packing = PackingType::InterleavedRow;
    out.is_complex = false;
    out.cts.push_back(std::move(output));
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_pre_compute_back_class_1(FHEAdapterContext &adapter,
                                                       const FHEBlockTensor &y_lr,
                                                       const FHEBlockTensor &W2,
                                                       const FHEBlockTensor &tanh_classifier,
                                                       const ClassifierBackwardOpParams &params,
                                                       const std::string &out_name,
                                                       const std::string &step_name = "pre_compute_back_class_1")
{
    
    if (params.batch_size <= 0 || params.row_X <= 0 || params.col_W <= 0 ||
        params.pre_blocks <= 0)
    {
        throw std::runtime_error(step_name + " requires positive classifier backward parameters");
    }
    require_single_ciphertext(y_lr, step_name);
    require_single_ciphertext(W2, step_name);
    require_single_ciphertext(tanh_classifier, step_name);
    if (y_lr.is_complex || W2.is_complex || tanh_classifier.is_complex)
    {
        throw std::runtime_error(step_name + " requires real packed classifier tensors");
    }

    PhantomCiphertext tanh_for_gate = tanh_classifier.cts.front();
    const int tanh_level = get_level(adapter.level_map, tanh_for_gate);
    if (tanh_level <= 0)
    {
        throw std::runtime_error(step_name + " cannot square tanh input at logical level " +
                                 std::to_string(tanh_level));
    }

    PhantomCiphertext gate;
    adapter.require_evaluator().square(tanh_for_gate, gate);
    adapter.require_evaluator().relinearize_inplace(gate, adapter.require_relin_keys());
    adapter.require_evaluator().rescale_to_next_inplace(gate);
    adapter.require_evaluator().negate_inplace(gate);
    PhantomPlaintext one = encode_real_constant_like(adapter, gate, 1.0);
    adapter.require_evaluator().add_plain_inplace(gate, one);

    PhantomCiphertext y = y_lr.cts.front();
    PhantomCiphertext w = W2.cts.front();
    const int y_w_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, y, w, y_lr.name + "/ct_0", W2.name + "/ct_0");
    if (y_w_level <= 0)
    {
        throw std::runtime_error(step_name + " cannot multiply y_lr and W2 at logical level " +
                                 std::to_string(y_w_level));
    }

    PhantomCiphertext weighted;
    adapter.require_evaluator().multiply(y, w, weighted);
    adapter.require_evaluator().relinearize_inplace(weighted, adapter.require_relin_keys());
    adapter.require_evaluator().rescale_to_next_inplace(weighted);

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    PhantomCiphertext rotated_weighted;
    adapter.require_evaluator().rotate_vector(weighted,
                                              n / 2,
                                              adapter.require_galois_keys(),
                                              rotated_weighted);
    adapter.require_evaluator().add_inplace(weighted, rotated_weighted);

    const int gate_weight_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, weighted, gate, step_name + "/weighted_y_w2", step_name + "/one_minus_tanh2");
    if (gate_weight_level <= 0)
    {
        throw std::runtime_error(step_name + " cannot multiply weighted output by tanh gate at logical level " +
                                 std::to_string(gate_weight_level));
    }
    
    adapter.require_evaluator().multiply_inplace(weighted, gate);
    adapter.require_evaluator().relinearize_inplace(weighted, adapter.require_relin_keys());
    adapter.require_evaluator().rescale_to_next_inplace(weighted);

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = tanh_classifier.logical_shape.empty()
                            ? std::vector<int>{1, params.col_W}
                            : tanh_classifier.logical_shape;
    out.packing = PackingType::InterleavedRow;
    out.is_complex = false;
    if (!params.pre_split_output)
    {
        out.cts.push_back(std::move(weighted));
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
        
        return out;
    }

    const int weighted_level = get_level(adapter.level_map, weighted);
    if (weighted_level <= 0)
    {
        throw std::runtime_error(step_name + " CPU split requires at least level 1 before mask rescale, got " +
                                 std::to_string(weighted_level));
    }
    if (params.pre_blocks * params.row_X * params.batch_size > n / 2)
    {
        throw std::runtime_error(step_name + " split masks exceed first half of CKKS slots");
    }

    out.cts.resize(static_cast<size_t>(params.pre_blocks));
    for (int i = 0; i < params.pre_blocks; ++i)
    {
        std::vector<double> mask(static_cast<size_t>(n), 0.0);
        for (int j = i * params.row_X * params.batch_size;
             j < (i + 1) * params.row_X * params.batch_size;
             ++j)
        {
            mask[static_cast<size_t>(j)] = 1.0;
            mask[static_cast<size_t>(j + n / 2)] = 1.0;
        }
        PhantomPlaintext encoded_mask;
        encode_values(adapter,
                          mask,
                          weighted.params_id(),
                          weighted.scale(),
                          encoded_mask);
        adapter.require_evaluator().multiply_plain(weighted, encoded_mask, out.cts[static_cast<size_t>(i)]);
        adapter.require_evaluator().rescale_to_next_inplace(out.cts[static_cast<size_t>(i)]);

        int rot_index = params.row_X * params.batch_size;
        while (rot_index <= n / 4)
        {
            PhantomCiphertext tmp2;
            adapter.require_evaluator().rotate_vector(out.cts[static_cast<size_t>(i)],
                                                      rot_index,
                                                      adapter.require_galois_keys(),
                                                      tmp2);
            adapter.require_evaluator().add_inplace(out.cts[static_cast<size_t>(i)], tmp2);
            
            rot_index *= 2;
        }
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_back_class_1_batch(FHEAdapterContext &adapter,
                                                 const FHEBlockTensor &ct,
                                                 const FHEBlockTensor &pre_compute_ct,
                                                 const ClassifierBackwardOpParams &params,
                                                 const std::string &out_name,
                                                 const std::string &step_name = "back_class_1_batch")
{
    
    if (params.batch_size <= 0 || params.row_X <= 0 || params.col_W <= 0 ||
        params.pre_blocks <= 0 || params.column_blocks <= 0)
    {
        throw std::runtime_error(step_name + " requires positive classifier backward parameters");
    }
    if (static_cast<int>(pre_compute_ct.cts.size()) != params.pre_blocks)
    {
        throw std::runtime_error(step_name + " pre_compute_ct count mismatch");
    }

    FHEBlockTensor ext_ct = phantom_ext_copy_x(adapter,
                                               ct,
                                               params.batch_size,
                                               params.row_X,
                                               out_name + "_ext_X",
                                               step_name + "/ext_copy_X");
    const int output_count = params.pre_blocks * params.column_blocks;
    if (static_cast<int>(ext_ct.cts.size()) != output_count)
    {
        if (static_cast<int>(ext_ct.cts.size()) != params.column_blocks)
        {
            throw std::runtime_error(step_name + " ext_ct count must be either column_blocks or pre_blocks*column_blocks");
        }
        
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = {params.col_W, params.col_W};
    out.packing = PackingType::RepeatedColumn;
    out.is_complex = false;
    out.cts.resize(static_cast<size_t>(output_count));

    for (int i = 0; i < params.pre_blocks; ++i)
    {
        for (int j = 0; j < params.column_blocks; ++j)
        {
            const int out_index = i * params.column_blocks + j;
            const int ext_index = static_cast<int>(ext_ct.cts.size()) == output_count
                                      ? out_index
                                      : j;
            PhantomCiphertext lhs = ext_ct.cts[static_cast<size_t>(ext_index)];
            PhantomCiphertext rhs = pre_compute_ct.cts[static_cast<size_t>(i)];
            const int start_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, ext_ct.name + "/ct_" + std::to_string(ext_index), pre_compute_ct.name + "/ct_" + std::to_string(i));
            if (start_level <= 0)
            {
                throw std::runtime_error(step_name + " cannot multiply at logical level " +
                                         std::to_string(start_level));
            }
            adapter.require_evaluator().multiply(lhs, rhs, out.cts[static_cast<size_t>(out_index)]);
            adapter.require_evaluator().relinearize_inplace(out.cts[static_cast<size_t>(out_index)],
                                                            adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(out.cts[static_cast<size_t>(out_index)]);
            
        }
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_gradient_x2(FHEAdapterContext &adapter,
                                          const FHEBlockTensor &W1,
                                          const FHEBlockTensor &pre_compute_ct,
                                          const ClassifierBackwardOpParams &params,
                                          const std::string &out_name,
                                          const std::string &step_name = "gradient_x2")
{
    
    if (params.batch_size <= 0 || params.row_X <= 0 || params.pre_blocks <= 0 ||
        params.column_blocks <= 0)
    {
        throw std::runtime_error(step_name + " requires positive classifier backward parameters");
    }
    if (static_cast<int>(pre_compute_ct.cts.size()) != params.pre_blocks ||
        static_cast<int>(W1.cts.size()) < params.pre_blocks * params.column_blocks)
    {
        throw std::runtime_error(step_name + " input ciphertext count mismatch");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    if (n % (params.row_X * params.batch_size) != 0)
    {
        throw std::runtime_error(step_name + " row_X*batch_size must divide slot count");
    }
    const int mask_chunks = n / (params.row_X * params.batch_size);
    std::vector<double> mask(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < mask_chunks; ++i)
    {
        for (int j = 0; j < params.batch_size; ++j)
        {
            mask[static_cast<size_t>(i * params.row_X * params.batch_size + j)] = 1.0;
        }
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = {params.row_X, params.col_W};
    out.packing = PackingType::InterleavedColumn;
    out.is_complex = false;
    out.cts.resize(static_cast<size_t>(params.column_blocks));

    for (int i = 0; i < params.column_blocks; ++i)
    {
        bool set = false;
        for (int j = 0; j < params.pre_blocks; ++j)
        {
             
            const int w_index = j * params.column_blocks + i;
            PhantomCiphertext lhs = W1.cts[static_cast<size_t>(w_index)];
            PhantomCiphertext rhs = pre_compute_ct.cts[static_cast<size_t>(j)];
            const int start_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, W1.name + "/ct_" + std::to_string(w_index), pre_compute_ct.name + "/ct_" + std::to_string(j));
            if (start_level <= 0)
            {
                throw std::runtime_error(step_name + " cannot multiply at logical level " +
                                         std::to_string(start_level));
            }
            PhantomCiphertext tmp;
            adapter.require_evaluator().multiply(lhs, rhs, tmp);
            
            if (!set)
            {
                out.cts[static_cast<size_t>(i)] = std::move(tmp);
                set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(out.cts[static_cast<size_t>(i)], tmp);
                
            }
        }
        adapter.require_evaluator().relinearize_inplace(out.cts[static_cast<size_t>(i)],
                                                        adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(out.cts[static_cast<size_t>(i)]);

        int rot_index = params.batch_size;
        while (rot_index < params.row_X * params.batch_size)
        {
            PhantomCiphertext tmp;
            adapter.require_evaluator().rotate_vector(out.cts[static_cast<size_t>(i)],
                                                      rot_index,
                                                      adapter.require_galois_keys(),
                                                      tmp);
            adapter.require_evaluator().add_inplace(out.cts[static_cast<size_t>(i)], tmp);
            
            rot_index *= 2;
        }

        PhantomPlaintext encoded_mask;
        encode_values(adapter,
                          mask,
                          out.cts[static_cast<size_t>(i)].params_id(),
                          out.cts[static_cast<size_t>(i)].scale(),
                          encoded_mask);
        adapter.require_evaluator().multiply_plain_inplace(out.cts[static_cast<size_t>(i)], encoded_mask);
        adapter.require_evaluator().rescale_to_next_inplace(out.cts[static_cast<size_t>(i)]);
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

}
