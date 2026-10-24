#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_backward_ops.cuh"
#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"
#include "source/fhe_transformer/fhe_packing_utils.cuh"

namespace fhe_transformer {

inline std::string backward_two_digit_index(int index)
{
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << index;
    return oss.str();
}

struct TransformerBackwardConfig {
    PhantomBootstrappingCallbacks callbacks;
    int gradient_ln2_input_level = 7;
    int ln2_no_norm_level = 6;
    int ln2_inv_sqrt_var_level = 4;
    int gradient_ln2_level = 3;
    int gradient_att_level = 0;
    int bts4_stored_level = 11;
    int gradient_ln1_level = 7;
    int ln1_no_norm_level = 10;
    int ln1_inv_sqrt_var_level = 8;
    int gradient_cat_level = 6;
    int gradient_u_level = 4;
    int d_level = 3;
    int gradient_qk_level = 0;
    int gradient_v_level = 0;
    int bts5_stored_level = 11;
    int gradient_a_input_level = 10;
    int gradient_a_level = 8;
    int gradient_b_input_level = 7;
    int gradient_b_level = 3;
    int gradient_x_level = 7;
    int num_heads = 12;

    bool write_cache = true;
    bool skip_lora_parameter_gradients = false;
    bool release_forward_cache_after_read = false;
    bool release_intermediates_after_use = false;
    LayerNormBackwardOpParams ln2_backward_params;
    LayerNormBackwardOpParams ln1_backward_params;
    GradientAttentionFFNOpParams gradient_att_params;
    MatrixOpParams gradient_cat_params;
    GradientUOpParams gradient_u_params;
    GradientQKOpParams gradient_qk_params;
    GradientVOpParams gradient_v_params;
    LoraGradientOpParams lora_gradient_params;
};

struct TransformerBackwardInputs {
    std::vector<std::vector<double>> W_o_plaintext;
    std::vector<std::vector<double>> W_Q_plaintext;
    std::vector<std::vector<double>> W_K_plaintext;
    std::vector<std::vector<double>> W_V_plaintext;
    std::vector<std::vector<double>> W_fc1_plaintext;
    std::vector<std::vector<double>> W_fc2_plaintext;
    FHEBlockTensor A_Q;
    FHEBlockTensor A_K;
    FHEBlockTensor A_V;
    FHEBlockTensor B_Q;
    FHEBlockTensor B_K;
    FHEBlockTensor B_V;
    FHEBlockTensor X_for_lora_grad;
};

struct TransformerBackwardResult {
    FHEBlockTensor gradient_LN2;
    FHEBlockTensor gradient_att_pre_bts;
    FHEBlockTensor gradient_att;
    FHEBlockTensor gradient_LN1;
    FHEBlockTensor gradient_cat;
    FHEBlockTensor gradient_U;
    FHEBlockTensor D;
    std::vector<FHEBlockTensor> gradient_Q_heads;
    std::vector<FHEBlockTensor> gradient_K_heads;
    std::vector<FHEBlockTensor> gradient_V_heads;
    FHEBlockTensor gradient_Q;
    FHEBlockTensor gradient_K;
    FHEBlockTensor gradient_V;
    FHEBlockTensor gradient_AQ;
    FHEBlockTensor gradient_AK;
    FHEBlockTensor gradient_AV;
    FHEBlockTensor gradient_BQ;
    FHEBlockTensor gradient_BK;
    FHEBlockTensor gradient_BV;
    FHEBlockTensor gradient_X;
};

inline void enable_cpu_style_transformer_backward_ffn_ln(TransformerBackwardConfig &config,
                                                         int batch_size,
                                                         int row,
                                                         int model_col,
                                                         int ffn_col)
{

    config.ln2_backward_params.batch_size = batch_size;
    config.ln2_backward_params.row = row;
    config.ln2_backward_params.col = model_col;
    config.ln2_backward_params.gamma = std::vector<double>(static_cast<size_t>(model_col), 1.0);

    config.ln1_backward_params.batch_size = batch_size;
    config.ln1_backward_params.row = row;
    config.ln1_backward_params.col = model_col;
    config.ln1_backward_params.gamma = std::vector<double>(static_cast<size_t>(model_col), 1.0);

    config.gradient_att_params.wf2_t_params.batch_size = batch_size;
    config.gradient_att_params.wf2_t_params.row_X = row;
    config.gradient_att_params.wf2_t_params.col_X = model_col;
    config.gradient_att_params.wf2_t_params.row_W = model_col;
    config.gradient_att_params.wf2_t_params.col_W = ffn_col;
    config.gradient_att_params.wf2_t_params.k = 1;

    config.gradient_att_params.wf1_t_params.batch_size = batch_size;
    config.gradient_att_params.wf1_t_params.row_X = row;
    config.gradient_att_params.wf1_t_params.col_X = ffn_col;
    config.gradient_att_params.wf1_t_params.row_W = ffn_col;
    config.gradient_att_params.wf1_t_params.col_W = model_col;
    config.gradient_att_params.wf1_t_params.k = 1;

    config.gradient_cat_params.batch_size = batch_size;
    config.gradient_cat_params.row_X = row;
    config.gradient_cat_params.col_X = model_col;
    config.gradient_cat_params.row_W = model_col;
    config.gradient_cat_params.col_W = model_col;
    config.gradient_cat_params.k = 1;
}

inline void enable_cpu_style_transformer_backward_attention_head(TransformerBackwardConfig &config,
                                                                int batch_size,
                                                                int row,
                                                                int head_dim,
                                                                int num_heads = 12)
{
    config.num_heads = num_heads;

    config.gradient_u_params.batch_size = batch_size;
    config.gradient_u_params.row = row;
    config.gradient_u_params.col = head_dim;

    config.gradient_qk_params.batch_size = batch_size;
    config.gradient_qk_params.row = row;
    config.gradient_qk_params.col = head_dim;
    config.gradient_qk_params.k = 1;

    config.gradient_v_params.batch_size = batch_size;
    config.gradient_v_params.row = row;
    config.gradient_v_params.col = head_dim;
    config.gradient_v_params.k = 1;
}

inline void enable_cpu_style_transformer_backward_lora_and_x(TransformerBackwardConfig &config,
                                                            int batch_size,
                                                            int row,
                                                            int model_col,
                                                            int rank)
{

    config.lora_gradient_params.batch_size = batch_size;
    config.lora_gradient_params.row = row;
    config.lora_gradient_params.col = model_col;
    config.lora_gradient_params.rank = rank;
    config.lora_gradient_params.k = 1;
}

inline FHEBlockTensor gradient_cat_head_for_attention(const PhantomContext &context, const FHELevelMap &level_map, FHEBlockTensor gradient_cat, const TransformerBackwardConfig &config, int head_index)
{
    const int head_dim = config.gradient_u_params.col;
    if (head_dim <= 0 || head_index < 0 || head_index >= config.num_heads)
    {
        throw std::runtime_error("gradient_cat_head_for_attention invalid head configuration");
    }
    if (gradient_cat.logical_shape.size() != 2)
    {
        throw std::runtime_error("gradient_cat_head_for_attention requires rank-2 gradient_cat");
    }

    if (gradient_cat.logical_shape[1] == head_dim)
    {
        gradient_cat.name = "gradient_cat_head_" + (head_index < 10 ? std::string("0") : std::string("")) +
                            std::to_string(head_index);

        return gradient_cat;
    }

    std::vector<FHEBlockTensor> heads = split_heads(gradient_cat, config.num_heads, head_dim);
    FHEBlockTensor head = heads.at(static_cast<size_t>(head_index));
    align_down_to(context, level_map, head, gradient_cat.level, head.name);
    return head;
}

inline FHEBlockTensor require_cached_level(const FHEForwardCache &cache, const std::string &key, const std::string &consumer, int level)
{
    const FHEBlockTensor &tensor = cache_read(cache, key, consumer, -1);
    if (tensor.level < level)
    {
        
        throw std::runtime_error(key + " insufficient level for " + consumer);
    }
    FHEBlockTensor copy = tensor;
    return copy;
}

inline FHEBlockTensor require_cached_level(FHEForwardCache &cache, const std::string &key, const std::string &consumer, int level, bool release_after_read)
{
    FHEBlockTensor copy =
        release_after_read ? cache_take(cache, key, consumer, -1)
                           : require_cached_level(static_cast<const FHEForwardCache &>(cache), key, consumer, level);
    if (copy.level < level)
    {
        
        throw std::runtime_error(key + " insufficient level for " + consumer);
    }
    return copy;
}

inline FHEBlockTensor bts5_refresh_head(FHEAdapterContext &adapter,
                                        const FHEBlockTensor &input,
                                        const TransformerBackwardConfig &config,
                                        const std::string &name)
{
    FHEBlockTensor refreshed = ordinary_bts(adapter,
                                            input,
                                            config.callbacks,
                                            name,
                                            config.bts5_stored_level,
                                            "BTS5_" + name);
    return refreshed;
}

inline TransformerBackwardResult transformer_backward(FHEAdapterContext &adapter,
                                                      const FHEBlockTensor &gradient_x2,
                                                      FHEForwardCache &forward_cache,
                                                      const TransformerBackwardInputs &inputs,
                                                      const TransformerBackwardConfig &config,
                                                      FHEBackwardCache &cache,
                                                      const std::string &block_name = "TransformerBackward")
{
    
    FHEBlockTensor grad_x2 = gradient_x2;
    align_down_to(adapter.require_context(), adapter.level_map, grad_x2, config.gradient_ln2_input_level, grad_x2.name);

    FHEBlockTensor ln2_no_norm =
        (require_cached_level(forward_cache, (forward_cache.records.count("ln2_normalized_input") ? "ln2_normalized_input" : "ln2_no_norm"), "gradient_LN2", config.ln2_no_norm_level, config.release_forward_cache_after_read));
    FHEBlockTensor ln2_inv =
        (require_cached_level(forward_cache, (forward_cache.records.count("ln2_inverse_stddev") ? "ln2_inverse_stddev" : "ln2_inv_sqrt_var"), "gradient_LN2", config.ln2_inv_sqrt_var_level, config.release_forward_cache_after_read));
    {
        align_down_to(adapter.require_context(), adapter.level_map, ln2_no_norm, config.ln2_no_norm_level, ln2_no_norm.name);
        align_down_to(adapter.require_context(), adapter.level_map, ln2_inv, config.ln2_inv_sqrt_var_level, ln2_inv.name);
    }

    TransformerBackwardResult result;
    {
        result.gradient_LN2 = phantom_gradient_layernorm(adapter,
                                                         grad_x2,
                                                         ln2_no_norm,
                                                         ln2_inv,
                                                         config.ln2_backward_params,
                                                         "gradient_LN2",
                                                         block_name + "_gradient_LN2");
        assert_level(result.gradient_LN2, config.gradient_ln2_level, "gradient_LN2");
    }

    FHEBlockTensor gelu_derivative =
        (require_cached_level(forward_cache, "gelu_derivative", "gradient_att", 4, config.release_forward_cache_after_read));
    FHEBlockTensor gelu_input =
        (require_cached_level(forward_cache, "gelu_input", "gradient_att", 5, config.release_forward_cache_after_read));
    {
        result.gradient_att_pre_bts = phantom_gradient_attention_ffn(adapter,
                                                                     result.gradient_LN2,
                                                                     gelu_derivative,
                                                                     gelu_input,
                                                                     inputs.W_fc1_plaintext,
                                                                     inputs.W_fc2_plaintext,
                                                                     config.gradient_att_params,
                                                                     "gradient_att_pre_BTS4",
                                                                     block_name + "_gradient_att");
        assert_level(result.gradient_att_pre_bts, config.gradient_att_level, "gradient_att_pre_BTS4");
    }
    if (config.release_intermediates_after_use)
    {
        grad_x2 = FHEBlockTensor{};
        ln2_no_norm = FHEBlockTensor{};
        ln2_inv = FHEBlockTensor{};
        result.gradient_LN2 = FHEBlockTensor{};
        gelu_derivative = FHEBlockTensor{};
        gelu_input = FHEBlockTensor{};
    }
    result.gradient_att = ordinary_bts(adapter,
                                       result.gradient_att_pre_bts,
                                       config.callbacks,
                                       "gradient_att",
                                       config.bts4_stored_level,
                                       block_name + "_BTS4");
    if (config.release_intermediates_after_use)
    {
        result.gradient_att_pre_bts = FHEBlockTensor{};
    }

    FHEBlockTensor ln1_no_norm =
        (require_cached_level(forward_cache, (forward_cache.records.count("ln1_normalized_input") ? "ln1_normalized_input" : "ln1_no_norm"), "gradient_LN1", config.ln1_no_norm_level, config.release_forward_cache_after_read));
    FHEBlockTensor ln1_inv =
        (require_cached_level(forward_cache, (forward_cache.records.count("ln1_inverse_stddev") ? "ln1_inverse_stddev" : "ln1_inv_sqrt_var"), "gradient_LN1", config.ln1_inv_sqrt_var_level, config.release_forward_cache_after_read));
    {
        result.gradient_LN1 = phantom_gradient_layernorm(adapter,
                                                         result.gradient_att,
                                                         ln1_no_norm,
                                                         ln1_inv,
                                                         config.ln1_backward_params,
                                                         "gradient_LN1",
                                                         block_name + "_gradient_LN1");
        assert_level(result.gradient_LN1, config.gradient_ln1_level, "gradient_LN1");
    }
    if (config.release_intermediates_after_use)
    {
        result.gradient_att = FHEBlockTensor{};
        ln1_no_norm = FHEBlockTensor{};
        ln1_inv = FHEBlockTensor{};
    }

    {
        result.gradient_cat = phantom_gradient_cat(adapter,
                                                   result.gradient_LN1,
                                                   inputs.W_o_plaintext,
                                                   config.gradient_cat_params,
                                                   "gradient_cat",
                                                   block_name + "_gradient_cat");
        assert_level(result.gradient_cat, config.gradient_cat_level, "gradient_cat");
    }

    {
        std::vector<FHEBlockTensor> q_heads;
        std::vector<FHEBlockTensor> k_heads;
        std::vector<FHEBlockTensor> v_heads;
        q_heads.reserve(static_cast<size_t>(config.num_heads));
        k_heads.reserve(static_cast<size_t>(config.num_heads));
        v_heads.reserve(static_cast<size_t>(config.num_heads));

        for (int head = 0; head < config.num_heads; ++head)
        {
            const std::string suffix = backward_two_digit_index(head);
            FHEBlockTensor gradient_cat_head = gradient_cat_head_for_attention(adapter.require_context(), adapter.level_map, result.gradient_cat, config, head);

            FHEBlockTensor v_head = require_cached_level(forward_cache, "V_head_" + suffix, "gradient_U", 6, config.release_forward_cache_after_read);
            FHEBlockTensor gradient_u = phantom_gradient_u(adapter,
                                                           gradient_cat_head,
                                                           v_head,
                                                           config.gradient_u_params,
                                                           "gradient_U_head_" + suffix,
                                                           block_name + "_gradient_U/head_" + suffix);
            assert_level(gradient_u, config.gradient_u_level, "gradient_U_head_" + suffix);
            if (head == 0)
            {
                result.gradient_U = gradient_u;
            }

            FHEBlockTensor ext_qk = require_cached_level(forward_cache, "ext_QK_head_" + suffix, "D", config.gradient_u_level, config.release_forward_cache_after_read);
            align_down_to(adapter.require_context(), adapter.level_map, ext_qk, config.gradient_u_level, ext_qk.name);
            FHEBlockTensor d_head = fhe_hadamard(adapter,
                                                 gradient_u,
                                                 ext_qk,
                                                 "D_head_" + suffix,
                                                 block_name + "_D_hadamard/head_" + suffix);
            assert_level(d_head, config.d_level, "D_head_" + suffix);
            if (head == 0)
            {
                result.D = d_head;
            }

            FHEBlockTensor q_head = require_cached_level(forward_cache, "Q_head_" + suffix, "gradient_Q", 2, config.release_forward_cache_after_read);
            FHEBlockTensor k_head = require_cached_level(forward_cache, "K_head_" + suffix, "gradient_K", 2, config.release_forward_cache_after_read);
            FHEBlockTensor rot_k = require_cached_level(forward_cache, "rot_K_head_" + suffix, "gradient_Q", 3, config.release_forward_cache_after_read);
            FHEBlockTensor real_q = require_cached_level(forward_cache, "real_Q_head_" + suffix, "gradient_K", 3, config.release_forward_cache_after_read);
            const int d_level = config.gradient_qk_level + 3;
            const int qk_value_level =
                config.gradient_qk_level + ((q_head.is_complex || k_head.is_complex) ? 2 : 1);
            align_down_to(adapter.require_context(), adapter.level_map, q_head, qk_value_level, q_head.name);
            align_down_to(adapter.require_context(), adapter.level_map, k_head, qk_value_level, k_head.name);
            align_down_to(adapter.require_context(), adapter.level_map, rot_k, d_level, rot_k.name);
            align_down_to(adapter.require_context(), adapter.level_map, real_q, d_level, real_q.name);
            FHEBlockTensor d_for_qk = d_head;
            align_down_to(adapter.require_context(), adapter.level_map, d_for_qk, d_level, d_for_qk.name);
            FHEBlockTensor gradient_q_pre_bts = phantom_gradient_q(adapter,
                                                                   d_for_qk,
                                                                   q_head,
                                                                   rot_k,
                                                                   config.gradient_qk_params,
                                                                   "gradient_Q_pre_BTS5_head_" + suffix,
                                                                   block_name + "_gradient_Q/head_" + suffix);
            FHEBlockTensor gradient_k_pre_bts = phantom_gradient_k(adapter,
                                                                   d_for_qk,
                                                                   k_head,
                                                                   real_q,
                                                                   config.gradient_qk_params,
                                                                   "gradient_K_pre_BTS5_head_" + suffix,
                                                                   block_name + "_gradient_K/head_" + suffix);
            assert_level(gradient_q_pre_bts, config.gradient_qk_level, "gradient_Q_pre_BTS5_head_" + suffix);
            assert_level(gradient_k_pre_bts, config.gradient_qk_level, "gradient_K_pre_BTS5_head_" + suffix);

            FHEBlockTensor ext_u = require_cached_level(forward_cache, "ext_U_head_" + suffix, "gradient_V", 1, config.release_forward_cache_after_read);
            FHEBlockTensor gradient_cat_for_v = gradient_cat_head;
            const int ext_u_level = config.gradient_v_level + 1;
            const int gradient_cat_level = config.gradient_v_level + 2;
            align_down_to(adapter.require_context(), adapter.level_map, gradient_cat_for_v, gradient_cat_level, gradient_cat_for_v.name);
            align_down_to(adapter.require_context(), adapter.level_map, ext_u, ext_u_level, ext_u.name);
            FHEBlockTensor gradient_v_pre_bts = phantom_gradient_v(adapter,
                                                                   gradient_cat_for_v,
                                                                   ext_u,
                                                                   config.gradient_v_params,
                                                                   "gradient_V_pre_BTS5_head_" + suffix,
                                                                   block_name + "_gradient_V/head_" + suffix);
            assert_level(gradient_v_pre_bts, config.gradient_v_level, "gradient_V_pre_BTS5_head_" + suffix);

            q_heads.push_back(bts5_refresh_head(adapter, gradient_q_pre_bts, config, "gradient_Q_head_" + suffix));
            k_heads.push_back(bts5_refresh_head(adapter, gradient_k_pre_bts, config, "gradient_K_head_" + suffix));
            v_heads.push_back(bts5_refresh_head(adapter, gradient_v_pre_bts, config, "gradient_V_head_" + suffix));
        }

        result.gradient_Q_heads = q_heads;
        result.gradient_K_heads = k_heads;
        result.gradient_V_heads = v_heads;
        result.gradient_Q = merge_heads(adapter.require_context(), adapter.level_map, q_heads, "gradient_Q");
        result.gradient_K = merge_heads(adapter.require_context(), adapter.level_map, k_heads, "gradient_K");
        result.gradient_V = merge_heads(adapter.require_context(), adapter.level_map, v_heads, "gradient_V");
        assert_level(result.gradient_Q, config.bts5_stored_level, "gradient_Q");
        assert_level(result.gradient_K, config.bts5_stored_level, "gradient_K");
        assert_level(result.gradient_V, config.bts5_stored_level, "gradient_V");
        if (config.release_intermediates_after_use)
        {
            q_heads.clear();
            k_heads.clear();
            v_heads.clear();
            result.gradient_Q_heads.clear();
            result.gradient_K_heads.clear();
            result.gradient_V_heads.clear();
        }
    }
    if (config.release_intermediates_after_use)
    {
        result.gradient_cat = FHEBlockTensor{};
        result.gradient_U = FHEBlockTensor{};
        result.D = FHEBlockTensor{};
    }

    if (!config.skip_lora_parameter_gradients)
    {
    FHEBlockTensor x_for_a = (inputs.X_for_lora_grad);
    align_down_to(adapter.require_context(), adapter.level_map, x_for_a, config.gradient_a_input_level, x_for_a.name);
    FHEBlockTensor gq_for_a = (result.gradient_Q);
    FHEBlockTensor gk_for_a = (result.gradient_K);
    FHEBlockTensor gv_for_a = (result.gradient_V);
    align_down_to(adapter.require_context(), adapter.level_map, gq_for_a, config.gradient_a_input_level, gq_for_a.name);
    align_down_to(adapter.require_context(), adapter.level_map, gk_for_a, config.gradient_a_input_level, gk_for_a.name);
    align_down_to(adapter.require_context(), adapter.level_map, gv_for_a, config.gradient_a_input_level, gv_for_a.name);
    {
        result.gradient_AQ = phantom_gradient_lora_a(adapter, x_for_a, gq_for_a, inputs.B_Q,
                                                     config.lora_gradient_params,
                                                     "gradient_AQ",
                                                     block_name + "_gradient_AQ");
        result.gradient_AK = phantom_gradient_lora_a(adapter, x_for_a, gk_for_a, inputs.B_K,
                                                     config.lora_gradient_params,
                                                     "gradient_AK",
                                                     block_name + "_gradient_AK");
        result.gradient_AV = phantom_gradient_lora_a(adapter, x_for_a, gv_for_a, inputs.B_V,
                                                     config.lora_gradient_params,
                                                     "gradient_AV",
                                                     block_name + "_gradient_AV");
    }
    if (config.release_intermediates_after_use)
    {
        x_for_a = FHEBlockTensor{};
        gq_for_a = FHEBlockTensor{};
        gk_for_a = FHEBlockTensor{};
        gv_for_a = FHEBlockTensor{};
    }

    FHEBlockTensor xa_q = require_cached_level(forward_cache, "XA_Q_cache", "gradient_BQ", config.gradient_b_input_level, config.release_forward_cache_after_read);
    FHEBlockTensor xa_k = require_cached_level(forward_cache, "XA_K_cache", "gradient_BK", config.gradient_b_input_level, config.release_forward_cache_after_read);
    FHEBlockTensor xa_v = require_cached_level(forward_cache, "XA_V_cache", "gradient_BV", config.gradient_b_input_level, config.release_forward_cache_after_read);
    FHEBlockTensor gq_for_b = (result.gradient_Q);
    FHEBlockTensor gk_for_b = (result.gradient_K);
    FHEBlockTensor gv_for_b = (result.gradient_V);
    align_down_to(adapter.require_context(), adapter.level_map, xa_q, config.gradient_b_input_level, xa_q.name);
    align_down_to(adapter.require_context(), adapter.level_map, xa_k, config.gradient_b_input_level, xa_k.name);
    align_down_to(adapter.require_context(), adapter.level_map, xa_v, config.gradient_b_input_level, xa_v.name);
    align_down_to(adapter.require_context(), adapter.level_map, gq_for_b, config.gradient_b_input_level, gq_for_b.name);
    align_down_to(adapter.require_context(), adapter.level_map, gk_for_b, config.gradient_b_input_level, gk_for_b.name);
    align_down_to(adapter.require_context(), adapter.level_map, gv_for_b, config.gradient_b_input_level, gv_for_b.name);
    {
        result.gradient_BQ = phantom_gradient_lora_b(adapter, xa_q, gq_for_b,
                                                     config.lora_gradient_params,
                                                     "gradient_BQ",
                                                     block_name + "_gradient_BQ",
                                                     inputs.B_Q.packing);
        result.gradient_BK = phantom_gradient_lora_b(adapter, xa_k, gk_for_b,
                                                     config.lora_gradient_params,
                                                     "gradient_BK",
                                                     block_name + "_gradient_BK",
                                                     inputs.B_K.packing);
        result.gradient_BV = phantom_gradient_lora_b(adapter, xa_v, gv_for_b,
                                                     config.lora_gradient_params,
                                                     "gradient_BV",
                                                     block_name + "_gradient_BV",
                                                     inputs.B_V.packing);
    }
    if (config.release_intermediates_after_use)
    {
        xa_q = FHEBlockTensor{};
        xa_k = FHEBlockTensor{};
        xa_v = FHEBlockTensor{};
        gq_for_b = FHEBlockTensor{};
        gk_for_b = FHEBlockTensor{};
        gv_for_b = FHEBlockTensor{};
    }
    }
    else
    {
        
    }

    {
        FHEBlockTensor grad_x_q = phantom_gradient_x_part1(adapter,
                                                           result.gradient_Q,
                                                           inputs.A_Q,
                                                           inputs.B_Q,
                                                           inputs.W_Q_plaintext,
                                                           config.lora_gradient_params,
                                                           "gradient_X_Q_branch",
                                                           block_name + "_gradient_X_Q_branch");
        FHEBlockTensor grad_x_k = phantom_gradient_x_part1(adapter,
                                                           result.gradient_K,
                                                           inputs.A_K,
                                                           inputs.B_K,
                                                           inputs.W_K_plaintext,
                                                           config.lora_gradient_params,
                                                           "gradient_X_K_branch",
                                                           block_name + "_gradient_X_K_branch");
        FHEBlockTensor grad_x_v = phantom_gradient_x_part1(adapter,
                                                           result.gradient_V,
                                                           inputs.A_V,
                                                           inputs.B_V,
                                                           inputs.W_V_plaintext,
                                                           config.lora_gradient_params,
                                                           "gradient_X_V_branch",
                                                           block_name + "_gradient_X_V_branch");
        result.gradient_X = phantom_gradient_x_sum(adapter,
                                                   result.gradient_LN1,
                                                   {grad_x_q, grad_x_k, grad_x_v},
                                                   "gradient_X",
                                                   block_name + "_gradient_X_decomposed_sum");
        if (config.release_intermediates_after_use)
        {
            grad_x_q = FHEBlockTensor{};
            grad_x_k = FHEBlockTensor{};
            grad_x_v = FHEBlockTensor{};
        }
    }
    if (config.release_intermediates_after_use)
    {
        result.gradient_LN1 = FHEBlockTensor{};
        result.gradient_Q = FHEBlockTensor{};
        result.gradient_K = FHEBlockTensor{};
        result.gradient_V = FHEBlockTensor{};
        forward_cache = FHEForwardCache{};
    }

    if (config.write_cache)
    {
        cache.gradient_AQ = result.gradient_AQ;
        cache.gradient_AK = result.gradient_AK;
        cache.gradient_AV = result.gradient_AV;
        cache.gradient_BQ = result.gradient_BQ;
        cache.gradient_BK = result.gradient_BK;
        cache.gradient_BV = result.gradient_BV;
        cache.gradient_X = result.gradient_X;
        if (!result.gradient_AQ.empty())
        {
            cache_write(cache, "gradient_AQ", result.gradient_AQ, "LoRA A_Q gradient", "AdamW A_Q");
            cache_write(cache, "gradient_AK", result.gradient_AK, "LoRA A_K gradient", "AdamW A_K");
            cache_write(cache, "gradient_AV", result.gradient_AV, "LoRA A_V gradient", "AdamW A_V");
        }
        if (!result.gradient_BQ.empty())
        {
            cache_write(cache, "gradient_BQ", result.gradient_BQ, "LoRA B_Q gradient", "AdamW B_Q");
            cache_write(cache, "gradient_BK", result.gradient_BK, "LoRA B_K gradient", "AdamW B_K");
            cache_write(cache, "gradient_BV", result.gradient_BV, "LoRA B_V gradient", "AdamW B_V");
        }
        cache_write(cache, "gradient_X", result.gradient_X, "decomposed transformer input gradient", "previous layer");
    }
    return result;
}

}
