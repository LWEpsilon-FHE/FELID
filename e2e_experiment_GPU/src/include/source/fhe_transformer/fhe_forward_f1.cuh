#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/fhe_matrix_ops.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

struct ForwardF1Config {
    MatrixOpParams qkv_pcmm_params;
    MatrixOpParams lora_a_params;
    MatrixOpParams lora_b_params;
    PhantomBootstrappingCallbacks callbacks;
    std::vector<double> bias_Q, bias_K, bias_V;
    int xw_qk_input_level = 8;
    int xw_qk_output_level = 7;
    int xw_v_input_level = 12;
    int xw_v_output_level = 11;
    int lora_qk_xa_cache_level = 11;
    int lora_qk_b_level = 8;
    int lora_qk_output_level = 7;
    int lora_v_input_level = 1;
    int lora_v_xa_level = 0;
    int lora_v_bts_stored_level = 12;
    int lora_v_output_level = 11;
    double lora_v_bootstrap_gain = 1.0;
};

struct ForwardF1Result {
    FHEBlockTensor XW_Q;
    FHEBlockTensor XW_K;
    FHEBlockTensor XW_V;
    FHEBlockTensor XA_Q_cache;
    FHEBlockTensor XA_K_cache;
    FHEBlockTensor XA_V_pre_bts;
    FHEBlockTensor XA_V_cache;
    FHEBlockTensor LoRA_Q;
    FHEBlockTensor LoRA_K;
    FHEBlockTensor LoRA_V;
    FHEBlockTensor Q_complex;
    FHEBlockTensor K_complex;
    FHEBlockTensor V_complex;
};

inline FHEBlockTensor run_lora_qk_path(FHEAdapterContext &adapter,
                                       const FHEBlockTensor &X,
                                       const FHEBlockTensor &A,
                                       const FHEBlockTensor &B,
                                       const MatrixOpParams &a_params,
                                       const MatrixOpParams &b_params,
                                       const ForwardF1Config &config,
                                       const std::string &prefix,
                                       FHEBlockTensor *xa_cache)
{
    FHEBlockTensor xa_high = phantom_ccmm_lora_a(adapter,
                                                X,
                                                A,
                                                a_params,
                                                "XA_" + prefix + "_cache",
                                                "F1_CCMM_A_" + prefix + "_cache");
    assert_level(xa_high, config.lora_qk_xa_cache_level, xa_high.name);
    if (xa_cache != nullptr)
    {
        *xa_cache = xa_high;
    }

    FHEBlockTensor xa_for_b = xa_high;
    xa_for_b.name = "XA_" + prefix + "_for_B";
    align_down_to(adapter.require_context(), adapter.level_map, xa_for_b, config.lora_qk_b_level, xa_for_b.name);

    FHEBlockTensor b_for_mul = B;
    align_down_to(adapter.require_context(), adapter.level_map, b_for_mul, config.lora_qk_b_level, b_for_mul.name);
    FHEBlockTensor lora = phantom_ccmm_lora_b_complex(adapter,
                                                      xa_for_b,
                                                      b_for_mul,
                                                      b_params,
                                                      "LoRA_" + prefix,
                                                      "F1_CCMM_B_complex_" + prefix);
    assert_level(lora, config.lora_qk_output_level, lora.name);
    return lora;
}

inline FHEBlockTensor run_lora_v_path(FHEAdapterContext &adapter,
                                      const FHEBlockTensor &X,
                                      const FHEBlockTensor &A,
                                      const FHEBlockTensor &B,
                                      const MatrixOpParams &a_params,
                                      const MatrixOpParams &b_params,
                                      const ForwardF1Config &config,
                                      FHEBlockTensor *xa_pre_bts,
                                      FHEBlockTensor *xa_cache)
{
    FHEBlockTensor x_low = X;
    x_low.name = X.name + "_for_LoRA_V";
    align_down_to(adapter.require_context(), adapter.level_map, x_low, config.lora_v_input_level, x_low.name);
    FHEBlockTensor a_low = A;
    a_low.name = A.name + "_for_LoRA_V";
    if (config.lora_v_bootstrap_gain != 1.0) {
        for (auto &ct : a_low.cts) {
            PhantomPlaintext factor;
            encode_values(adapter, config.lora_v_bootstrap_gain, ct.params_id(), ct.scale(), factor);
            ct = multiply_plain_and_rescale(adapter, ct, factor, "LoRA_V_bootstrap_gain");
        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, a_low);
    }
    align_down_to(adapter.require_context(), adapter.level_map, a_low, config.lora_v_input_level, a_low.name);

    FHEBlockTensor xa = phantom_ccmm_lora_a(adapter,
                                           x_low,
                                           a_low,
                                           a_params,
                                           "XA_V_pre_bts",
                                           "F1_CCMM_A_V");
    assert_level(xa, config.lora_v_xa_level, xa.name);
    if (xa_pre_bts != nullptr)
    {
        *xa_pre_bts = xa;
    }

    auto callbacks = config.callbacks;
    if (config.lora_v_bootstrap_gain != 1.0) {
        if (!callbacks.gain_compensated_bts)
            throw std::runtime_error("LoRA V bootstrap gain requires compensated callbacks");
        callbacks.ordinary_bts = callbacks.gain_compensated_bts(config.lora_v_bootstrap_gain);
    }
    FHEBlockTensor xa_refreshed = ordinary_bts(adapter,
                                               xa,
                                               callbacks,
                                               "XA_V_after_BTS",
                                               config.lora_v_bts_stored_level,
                                               "F1_LoRA_V_ordinary_BTS");
    if (xa_cache != nullptr)
    {
        FHEBlockTensor xa_for_grad_b = xa_refreshed;
        xa_for_grad_b.name = "XA_V_cache";
        *xa_cache = xa_for_grad_b;
    }
    FHEBlockTensor b_top = B;
    assert_level(b_top, config.lora_v_bts_stored_level, b_top.name);
    FHEBlockTensor lora = phantom_ccmm_lora_b_complex(adapter,
                                                      xa_refreshed,
                                                      b_top,
                                                      b_params,
                                                      "LoRA_V",
                                                      "F1_CCMM_B_complex_V");
    assert_level(lora, config.lora_v_output_level, lora.name);
    return lora;
}

inline ForwardF1Result forward_block_f1_qkv_lora(FHEAdapterContext &adapter,
                                                 const FHEBlockTensor &X,
                                                 const std::vector<std::vector<double>> &W_Q,
                                                 const std::vector<std::vector<double>> &W_K,
                                                 const std::vector<std::vector<double>> &W_V,
                                                 const FHEBlockTensor &A_Q,
                                                 const FHEBlockTensor &B_Q,
                                                 const FHEBlockTensor &A_K,
                                                 const FHEBlockTensor &B_K,
                                                 const FHEBlockTensor &A_V,
                                                 const FHEBlockTensor &B_V,
                                                 const ForwardF1Config &config,
                                                 FHEForwardCache &cache,
                                                 const std::string &block_name = "F1")
{

    FHEBlockTensor x_qk = X;
    x_qk.name = X.name + "_for_XW_QK";
    align_down_to(adapter.require_context(), adapter.level_map, x_qk, config.xw_qk_input_level, x_qk.name);
    FHEBlockTensor x_v = X;
    x_v.name = X.name + "_for_XW_V";
    align_down_to(adapter.require_context(), adapter.level_map, x_v, config.xw_v_input_level, x_v.name);

    ForwardF1Result result;
    result.XW_Q = phantom_pcmm_complex(adapter, x_qk, W_Q, config.qkv_pcmm_params,
                                       "XW_Q", false, block_name + "_XW_Q_PCMM_complex");
    result.XW_K = phantom_pcmm_complex(adapter, x_qk, W_K, config.qkv_pcmm_params,
                                       "XW_K", false, block_name + "_XW_K_PCMM_complex");
    result.XW_V = phantom_pcmm_complex(adapter, x_v, W_V, config.qkv_pcmm_params,
                                       "XW_V", false, block_name + "_XW_V_PCMM_complex");
    add_column_dense_bias(adapter, result.XW_Q, config.bias_Q, config.qkv_pcmm_params);
    add_column_dense_bias(adapter, result.XW_K, config.bias_K, config.qkv_pcmm_params);
    add_column_dense_bias(adapter, result.XW_V, config.bias_V, config.qkv_pcmm_params);
    assert_level(result.XW_Q, config.xw_qk_output_level, "XW_Q");
    assert_level(result.XW_K, config.xw_qk_output_level, "XW_K");
    assert_level(result.XW_V, config.xw_v_output_level, "XW_V");

    result.LoRA_Q = run_lora_qk_path(adapter, X, A_Q, B_Q,
                                     config.lora_a_params, config.lora_b_params,
                                     config, "Q", &result.XA_Q_cache);
    result.LoRA_K = run_lora_qk_path(adapter, X, A_K, B_K,
                                     config.lora_a_params, config.lora_b_params,
                                     config, "K", &result.XA_K_cache);
    result.LoRA_V = run_lora_v_path(adapter, X, A_V, B_V,
                                    config.lora_a_params, config.lora_b_params,
                                    config, &result.XA_V_pre_bts, &result.XA_V_cache);

    result.Q_complex = fhe_add(adapter, result.XW_Q, result.LoRA_Q, "Q_complex", block_name + "_Q_add");
    result.K_complex = fhe_add(adapter, result.XW_K, result.LoRA_K, "K_complex", block_name + "_K_add");
    result.V_complex = fhe_add(adapter, result.XW_V, result.LoRA_V, "V_complex", block_name + "_V_add");
    assert_level(result.Q_complex, config.xw_qk_output_level, "Q_complex");
    assert_level(result.K_complex, config.xw_qk_output_level, "K_complex");
    assert_level(result.V_complex, config.xw_v_output_level, "V_complex");

    cache_write(cache, "XA_Q_cache", result.XA_Q_cache, "LoRA Q XA high-level cache", "gradient_AQ");
    cache_write(cache, "XA_K_cache", result.XA_K_cache, "LoRA K XA high-level cache", "gradient_AK");
    cache_write(cache, "XA_V_cache", result.XA_V_cache, "LoRA V XA after ordinary BTS high-level cache", "gradient_BV");

    return result;
}

}
