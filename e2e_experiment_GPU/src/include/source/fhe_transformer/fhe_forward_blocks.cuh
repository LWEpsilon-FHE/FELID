#pragma once

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/fhe_gelu.cuh"
#include "source/fhe_transformer/fhe_layernorm.cuh"
#include "source/fhe_transformer/fhe_matrix_ops.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"
#include "source/fhe_transformer/fhe_packing_utils.cuh"

namespace fhe_transformer {

inline std::string two_digit_index(int index)
{
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << index;
    return oss.str();
}

struct ForwardF2Config {
    int num_heads = 12;
    int head_dim = 64;
    AttentionOpParams kernel_params;
    AttentionValueOpParams value_params;
    PhantomBootstrappingCallbacks callbacks;
    bool cache_backward_auxiliary = true;
    int qk_complex_input_level = 7;
    int v_complex_input_level = 11;
    int qk_real_level = 6;
    int kernel_level = 3;
    int exp_stored_level = 11;
    int h_level = 8;
};

struct ForwardF2Result {
    std::vector<FHEBlockTensor> Q_heads;
    std::vector<FHEBlockTensor> K_heads;
    std::vector<FHEBlockTensor> V_heads;
    std::vector<FHEBlockTensor> exp_kernel_heads;
    std::vector<FHEBlockTensor> H_heads;
    FHEBlockTensor H_merged;
};

struct ForwardF3Config {
    MatrixOpParams wo_params;
    int h_input_level = 8;
    int attention_real_level = 7;
    bool selfoutput_op_outputs_real = true;
    std::vector<double> bias;
};

struct ForwardLayerNormConfig {
    LayerNormParams params;
    LayerNormBackend backend;
};

struct ForwardF3Result {
    FHEBlockTensor attention_out_real;
    FHEBlockTensor X_after_attn_residual;
};

struct ForwardF4Config {
    MatrixOpParams fc1_params;
    MatrixOpParams fc2_params;
    GeluParams gelu_params;
    PhantomBootstrappingCallbacks gelu_callbacks;
    int ln1_input_level = 5;
    int fc1_complex_level = 4;
    int fc1_real_level = 3;
    int gelu_output_level = 4;
    int fc2_complex_level = 3;
    int fc2_real_level = 2;
    bool drop_non_cached_intermediates = false;
    std::vector<double> bias_fc1, bias_fc2;
};

struct ForwardF4Result {
    FHEBlockTensor fc1_complex;
    FHEBlockTensor fc1_real;
    FHEBlockTensor gelu_input;
    FHEBlockTensor gelu_derivative;
    FHEBlockTensor gelu_output;
    FHEBlockTensor fc2_complex;
    FHEBlockTensor fc2_real;
    FHEBlockTensor X_after_ffn_residual;
};

struct ForwardF4LayerNormBtsConfig {
    ForwardLayerNormConfig ln2_config;
    PhantomBootstrappingCallbacks bts_callbacks;
    int ln2_input_level = 2;
    int ln2_output_level = 0;
    int bts_stored_level = 12;
};

struct ForwardF4LayerNormBtsResult {
    LayerNormResult ln2;
    FHEBlockTensor X_after_ln2;
};

inline void write_forward_cache(FHEForwardCache &cache, const std::string &key, const FHEBlockTensor &tensor, const std::string &why_cached, const std::string &consumer)
{
    cache_write(cache, key, tensor, why_cached, consumer);
}

inline ForwardF2Result forward_block_f2_attention(FHEAdapterContext &adapter,
                                                  const FHEBlockTensor &Q_complex,
                                                  const FHEBlockTensor &K_complex,
                                                  const FHEBlockTensor &V_complex,
                                                  const ForwardF2Config &config,
                                                  FHEForwardCache &cache,
                                                  const std::string &block_name = "F2")
{

    assert_level(Q_complex, config.qk_complex_input_level, "Q_complex");
    assert_level(K_complex, config.qk_complex_input_level, "K_complex");
    assert_level(V_complex, config.v_complex_input_level, "V_complex");

    FHEBlockTensor Q_real = convert_complex_to_real(adapter,
                                                    Q_complex,
                                                    "Q_real",
                                                    1.0,
                                                    block_name + "_Q_complex_to_real");
    FHEBlockTensor K_real = convert_complex_to_real(adapter,
                                                    K_complex,
                                                    "K_real",
                                                    1.0,
                                                    block_name + "_K_complex_to_real");
    assert_level(Q_real, config.qk_real_level, "Q_real");
    assert_level(K_real, config.qk_real_level, "K_real");

    ForwardF2Result result;
    result.Q_heads = split_heads(Q_real, config.num_heads, config.head_dim);
    result.K_heads = split_heads(K_real, config.num_heads, config.head_dim);
    result.V_heads = split_heads(V_complex, config.num_heads, config.head_dim);
    for (int head = 0; head < config.num_heads; ++head)
    {
        const std::string suffix = two_digit_index(head);
        if (config.cache_backward_auxiliary)
        {
            write_forward_cache(cache, "Q_head_" + suffix, result.Q_heads.at(head), "forward attention Q head", "gradient_Q");
            write_forward_cache(cache, "K_head_" + suffix, result.K_heads.at(head), "forward attention K head", "gradient_K");
            write_forward_cache(cache, "V_head_" + suffix, result.V_heads.at(head), "forward attention V head", "gradient_V");
            FHEBlockTensor real_q_cache = real_Q(result.Q_heads.at(head), "real_Q_head_" + suffix);
            FHEBlockTensor rot_k_cache = phantom_rot_k_for_gradient_q(adapter,
                                                                      result.K_heads.at(head),
                                                                      config.kernel_params,
                                                                      "rot_K_head_" + suffix,
                                                                      block_name + "_rot_K_cache/head_" + suffix);
            write_forward_cache(cache, "real_Q_head_" + suffix, real_q_cache, "real Q cache for gradient_K", "gradient_K");
            write_forward_cache(cache, "rot_K_head_" + suffix, rot_k_cache, "rotated K cache for gradient_Q", "gradient_Q");
        }

        AttentionOpParams scored_kernel_params = config.kernel_params;

        scored_kernel_params.distance_scale = -0.5 / std::sqrt(static_cast<double>(config.head_dim));
        FHEBlockTensor kernel = phantom_ccmm_kernel_bsgs(adapter,
                                                         result.Q_heads.at(head),
                                                         result.K_heads.at(head),
                                                         scored_kernel_params,
                                                         "kernel_head_" + suffix,
                                                         block_name + "_attention_kernel/head_" + suffix);
        assert_level(kernel, config.kernel_level, "kernel_head_" + suffix);

        FHEBlockTensor exp_kernel = exp_fb(adapter,
                                           kernel,
                                           config.callbacks,
                                           "exp_kernel_head_" + suffix,
                                           config.kernel_level,
                                           config.exp_stored_level,
                                           block_name + "_attention_EXP/head_" + suffix);
        exp_kernel.packing = PackingType::InterleavedDiagonal;
        result.exp_kernel_heads.push_back(exp_kernel);
        if (config.cache_backward_auxiliary)
        {
            write_forward_cache(cache, "exp_kernel_head_" + suffix, exp_kernel, "EXP FB attention kernel", "gradient_U");
        }

        FHEBlockTensor ext_cache;
        FHEBlockTensor H_head = phantom_ccmm_v_bsgs_complex(adapter,
                                                            exp_kernel,
                                                            result.V_heads.at(head),
                                                            config.value_params,
                                                            "H_head_" + suffix,
                                                            config.cache_backward_auxiliary ? &ext_cache : nullptr,
                                                            block_name + "_attention_V/head_" + suffix);
        assert_level(H_head, config.h_level, "H_head_" + suffix);
        if (config.cache_backward_auxiliary)
        {
            FHEBlockTensor ext_u_cache = ext_cache;
            ext_u_cache.name = "ext_U_head_" + suffix;
            write_forward_cache(cache, "ext_QK_head_" + suffix, ext_cache, "expanded attention kernel for V multiplication", "gradient_D");
            write_forward_cache(cache, "ext_U_head_" + suffix, ext_u_cache, "expanded attention U cache for gradient_V", "gradient_V");
        }
        result.H_heads.push_back(H_head);
    }

    result.H_merged = merge_heads(adapter.require_context(), adapter.level_map, result.H_heads, "H_merged");
    assert_level(result.H_merged, config.h_level, "H_merged");
    return result;
}

inline ForwardF3Result forward_block_f3_attention_projection(FHEAdapterContext &adapter,
                                                             const FHEBlockTensor &H_merged,
                                                             const FHEBlockTensor &X_residual,
                                                             const std::vector<std::vector<double>> &W_o,
                                                             const ForwardF3Config &config,
                                                             FHEForwardCache &cache,
                                                             const std::string &block_name = "F3")
{

    assert_level(H_merged, config.h_input_level, "H_merged");

    ForwardF3Result result;
    if (config.selfoutput_op_outputs_real)
    {
        result.attention_out_real = phantom_pcmm_selfoutput_real(adapter,
                                                                 H_merged,
                                                                 W_o,
                                                                 config.wo_params,
                                                                 "attention_out_real",
                                                                 block_name + "_PCMM_selfoutput_complex");
    }
    else
    {
        FHEBlockTensor attention_out_complex = phantom_pcmm_complex(adapter,
                                                                    H_merged,
                                                                    W_o,
                                                                    config.wo_params,
                                                                    "attention_out_complex",
                                                                    true,
                                                                    block_name + "_PCMM_selfoutput_complex");
        result.attention_out_real = convert_complex_to_real(adapter,
                                                            attention_out_complex,
                                                            "attention_out_real",
                                                            1.0,
                                                            block_name + "_attention_complex_to_real");
    }
    add_column_dense_bias(adapter, result.attention_out_real, config.bias, config.wo_params);
    assert_level(result.attention_out_real, config.attention_real_level, "attention_out_real");
    if (X_residual.level < result.attention_out_real.level)
    {
        
        throw std::runtime_error("F3 residual add requires X_residual at attention_out_real level or higher");
    }

    result.X_after_attn_residual = fhe_add(adapter,
                                           X_residual,
                                           result.attention_out_real,
                                           "X_after_attn_residual",
                                           block_name + "_residual_add");
    assert_level(result.X_after_attn_residual, config.attention_real_level, "X_after_attn_residual");
    return result;
}

inline LayerNormResult forward_block_f3_layernorm1(FHEAdapterContext &adapter,
                                                   const FHEBlockTensor &X_after_attn_residual,
                                                   const ForwardLayerNormConfig &config,
                                                   FHEForwardCache &cache,
                                                   const std::string &block_name = "F3_LayerNorm1")
{
    LayerNormResult result = run_layernorm(adapter,
                                           X_after_attn_residual,
                                           config.params,
                                           config.backend,
                                           block_name);
    result.no_norm.name = "ln1_no_norm";
    result.inv_sqrt_var.name = "ln1_inv_sqrt_var";
    result.output.name = "X_after_ln1";

    write_forward_cache(cache, result.normalized_input.empty() ? "ln1_no_norm" : "ln1_normalized_input", result.normalized_input.empty() ? result.no_norm : result.normalized_input, "LayerNorm1 normalized input for backward", "gradient_LN1");
    write_forward_cache(cache, result.inverse_stddev.empty() ? "ln1_inv_sqrt_var" : "ln1_inverse_stddev", result.inverse_stddev.empty() ? result.inv_sqrt_var : result.inverse_stddev, "LayerNorm1 unscaled inverse stddev for backward", "gradient_LN1");
    return result;
}

inline ForwardF4Result forward_block_f4_ffn(FHEAdapterContext &adapter,
                                            const FHEBlockTensor &X_after_ln1,
                                            const FHEBlockTensor &X_after_attn_residual,
                                            const std::vector<std::vector<double>> &W_fc1,
                                            const std::vector<std::vector<double>> &W_fc2,
                                            const ForwardF4Config &config,
                                            FHEForwardCache &cache,
                                            const std::string &block_name = "F4")
{

    assert_level(X_after_ln1, config.ln1_input_level, "X_after_ln1");

    ForwardF4Result result;
    result.fc1_complex = phantom_pcmm_complex(adapter,
                                              X_after_ln1,
                                              W_fc1,
                                              config.fc1_params,
                                              "fc1_complex",
                                              false,
                                              block_name + "_fc1_PCMM_complex");
    add_column_dense_bias(adapter, result.fc1_complex, config.bias_fc1, config.fc1_params);
    assert_level(result.fc1_complex, config.fc1_complex_level, "fc1_complex");
    result.fc1_real = convert_complex_to_real(adapter,
                                              result.fc1_complex,
                                              "fc1_real",
                                              1.0,
                                              block_name + "_fc1_complex_to_real");
    assert_level(result.fc1_real, config.fc1_real_level, "fc1_real");
    if (config.drop_non_cached_intermediates)
    {
        result.fc1_complex = FHEBlockTensor{};
    }
    GeluResult gelu = run_gelu_with_rboot(adapter,
                                          result.fc1_real,
                                          config.gelu_params,
                                          config.gelu_callbacks,
                                          block_name + "_GELU_RBOOT");
    result.gelu_input = gelu.gelu_input;
    result.gelu_derivative = gelu.gelu_derivative;
    result.gelu_output = gelu.gelu_output;
    assert_level(result.gelu_output, config.gelu_output_level, "gelu_output");
    if (config.drop_non_cached_intermediates)
    {
        result.fc1_real = FHEBlockTensor{};
    }

    write_forward_cache(cache, "gelu_input", result.gelu_input, "GELU input retained for derivative", "FFN backward GELU derivative");
    write_forward_cache(cache, "gelu_derivative", result.gelu_derivative, "Function bootstrap GELU derivative", "FFN backward GELU derivative");
    result.fc2_complex = phantom_pcmm_complex(adapter,
                                              result.gelu_output,
                                              W_fc2,
                                              config.fc2_params,
                                              "fc2_complex",
                                              false,
                                              block_name + "_fc2_PCMM_complex");
    add_column_dense_bias(adapter, result.fc2_complex, config.bias_fc2, config.fc2_params);
    assert_level(result.fc2_complex, config.fc2_complex_level, "fc2_complex");
    if (config.drop_non_cached_intermediates)
    {
        result.gelu_output = FHEBlockTensor{};
    }
    result.fc2_real = convert_complex_to_real(adapter,
                                              result.fc2_complex,
                                              "fc2_real",
                                              1.0,
                                              block_name + "_fc2_complex_to_real");
    assert_level(result.fc2_real, config.fc2_real_level, "fc2_real");
    if (config.drop_non_cached_intermediates)
    {
        result.fc2_complex = FHEBlockTensor{};
    }
    if (X_after_attn_residual.level < result.fc2_real.level)
    {
        
        throw std::runtime_error("F4 residual add requires residual branch at fc2_real level or higher");
    }
    result.X_after_ffn_residual = fhe_add(adapter,
                                          X_after_attn_residual,
                                          result.fc2_real,
                                          "X_after_ffn_residual",
                                          block_name + "_residual_add");
    assert_level(result.X_after_ffn_residual, config.fc2_real_level, "X_after_ffn_residual");
    if (config.drop_non_cached_intermediates)
    {
        result.fc2_real = FHEBlockTensor{};
    }
    return result;
}

inline ForwardF4LayerNormBtsResult forward_block_f4_layernorm2_bts(
    FHEAdapterContext &adapter,
    const FHEBlockTensor &X_after_ffn_residual,
    const ForwardF4LayerNormBtsConfig &config,
    FHEForwardCache &cache,
    const std::string &block_name = "F4_LayerNorm2_BTS")
{
    
    assert_level(X_after_ffn_residual, config.ln2_input_level, "X_after_ffn_residual");

    ForwardLayerNormConfig ln2_config = config.ln2_config;
    ln2_config.params.expected_input_level = config.ln2_input_level;
    ln2_config.params.expected_output_level = config.ln2_output_level;

    ForwardF4LayerNormBtsResult result;
    result.ln2 = run_layernorm(adapter,
                               X_after_ffn_residual,
                               ln2_config.params,
                               ln2_config.backend,
                               block_name + "_LayerNorm2");
    result.ln2.no_norm.name = "ln2_no_norm";
    result.ln2.inv_sqrt_var.name = "ln2_inv_sqrt_var";
    result.ln2.output.name = "X_after_ln2_pre_bts";
    assert_level(result.ln2.output, config.ln2_output_level, result.ln2.output.name);

    write_forward_cache(cache, result.ln2.normalized_input.empty() ? "ln2_no_norm" : "ln2_normalized_input", result.ln2.normalized_input.empty() ? result.ln2.no_norm : result.ln2.normalized_input, "LayerNorm2 normalized input for backward", "gradient_LN2");
    write_forward_cache(cache, result.ln2.inverse_stddev.empty() ? "ln2_inv_sqrt_var" : "ln2_inverse_stddev", result.ln2.inverse_stddev.empty() ? result.ln2.inv_sqrt_var : result.ln2.inverse_stddev, "LayerNorm2 unscaled inverse stddev for backward", "gradient_LN2");

    auto output_callbacks=config.bts_callbacks;
    if (ln2_config.params.output_gain!=1) {
        if (!output_callbacks.gain_compensated_bts)
            throw std::runtime_error("Scaled LayerNorm output needs a compensated bootstrap");
        output_callbacks.ordinary_bts=output_callbacks.gain_compensated_bts(ln2_config.params.output_gain);
    }
    result.X_after_ln2 = ordinary_bts(adapter,
                                      result.ln2.output,
                                      output_callbacks,
                                      "X_after_ln2",
                                      config.bts_stored_level,
                                      block_name + "_BTS1");
    assert_level(result.X_after_ln2, config.bts_stored_level, "X_after_ln2");
    result.X_after_ln2.packing = PackingType::InterleavedColumn;
    result.X_after_ln2.is_complex = false;
    return result;
}

}
