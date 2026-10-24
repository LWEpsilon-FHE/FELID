#pragma once

#include <array>
#include <optional>
#include <vector>

#include "source/fhe_transformer/fhe_adamw.cuh"
#include "source/fhe_transformer/fhe_compact_optimizer.cuh"
#include "source/fhe_transformer/fhe_backward_blocks.cuh"
#include "source/fhe_transformer/fhe_classifier_blocks.cuh"
#include "source/fhe_transformer/fhe_forward_f1.cuh"
#include "source/fhe_transformer/fhe_forward_blocks.cuh"

namespace fhe_transformer {

struct PipelineConfigs {
    ForwardF1Config f1;
    ForwardF2Config f2;
    ForwardF3Config f3;
    ForwardLayerNormConfig ln1;
    ForwardF4Config f4;
    ForwardF4LayerNormBtsConfig ln2_bts;
    ClassifierForwardConfig classifier_forward;
    ClassifierBackwardConfig classifier_backward;
    TransformerBackwardConfig transformer_backward;
    AdamWHEConfig adamw;
};

struct LayerInputs {
    FHEBlockTensor X;
    std::vector<std::vector<double>> W_Q;
    std::vector<std::vector<double>> W_K;
    std::vector<std::vector<double>> W_V;
    std::vector<std::vector<double>> W_o;
    std::vector<std::vector<double>> W_fc1;
    std::vector<std::vector<double>> W_fc2;
    FHEBlockTensor A_Q;
    FHEBlockTensor B_Q;
    FHEBlockTensor A_K;
    FHEBlockTensor B_K;
    FHEBlockTensor A_V;
    FHEBlockTensor B_V;
    TransformerBackwardInputs backward_inputs;
};

struct LayerResult {
    FHEForwardCache forward_cache;
    FHEBackwardCache backward_cache;
    ForwardF1Result f1;
    ForwardF2Result f2;
    ForwardF3Result f3;
    LayerNormResult ln1;
    ForwardF4Result f4;
    ForwardF4LayerNormBtsResult ln2_bts;
    TransformerBackwardResult backward;
};

inline std::array<const FHEBlockTensor *,6> pipeline_lora_parameters(const LayerInputs &layer)
{
    return {&layer.A_Q,&layer.A_K,&layer.A_V,&layer.B_Q,&layer.B_K,&layer.B_V};
}

inline std::array<const FHEBlockTensor *,6> pipeline_lora_gradients(const TransformerBackwardResult &gradient)
{
    return {&gradient.gradient_AQ,&gradient.gradient_AK,&gradient.gradient_AV,
            &gradient.gradient_BQ,&gradient.gradient_BK,&gradient.gradient_BV};
}

inline FHEBlockTensor pipeline_unpack_parameter(FHEAdapterContext &adapter,
                                             const CompactOptimizerLayout &layout,
                                             const CompactOptimizerState &state, size_t matrix,
                                             const AdamWHEConfig &config)
{
    auto out=compact_unpack_matrix(adapter,layout,state.theta,matrix,
                                   "updated_trainable_"+std::to_string(matrix));
    if (matrix<12)
        out=adamw_refresh_to_input_level(adapter,out,config,out.name+"_forward_ready");
    return out;
}

template <typename LayerInputs>
inline void fill_transformer_backward_plaintexts(TransformerBackwardInputs &backward_inputs,
                                                 const LayerInputs &inputs)
{
    if (backward_inputs.W_o_plaintext.empty())
    {
        backward_inputs.W_o_plaintext = inputs.W_o;
    }
    if (backward_inputs.W_Q_plaintext.empty())
    {
        backward_inputs.W_Q_plaintext = inputs.W_Q;
    }
    if (backward_inputs.W_K_plaintext.empty())
    {
        backward_inputs.W_K_plaintext = inputs.W_K;
    }
    if (backward_inputs.W_V_plaintext.empty())
    {
        backward_inputs.W_V_plaintext = inputs.W_V;
    }
    if (backward_inputs.W_fc1_plaintext.empty())
    {
        backward_inputs.W_fc1_plaintext = inputs.W_fc1;
    }
    if (backward_inputs.W_fc2_plaintext.empty())
    {
        backward_inputs.W_fc2_plaintext = inputs.W_fc2;
    }
    if (backward_inputs.A_Q.empty())
    {
        backward_inputs.A_Q = inputs.A_Q;
    }
    if (backward_inputs.A_K.empty())
    {
        backward_inputs.A_K = inputs.A_K;
    }
    if (backward_inputs.A_V.empty())
    {
        backward_inputs.A_V = inputs.A_V;
    }
    if (backward_inputs.B_Q.empty())
    {
        backward_inputs.B_Q = inputs.B_Q;
    }
    if (backward_inputs.B_K.empty())
    {
        backward_inputs.B_K = inputs.B_K;
    }
    if (backward_inputs.B_V.empty())
    {
        backward_inputs.B_V = inputs.B_V;
    }
    if (backward_inputs.X_for_lora_grad.empty())
    {
        backward_inputs.X_for_lora_grad = inputs.X;
    }
}

}
