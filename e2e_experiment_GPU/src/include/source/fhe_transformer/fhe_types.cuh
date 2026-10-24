#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "phantom.h"

namespace fhe_transformer {

enum class PackingType {
    Unknown = 0,
    InterleavedColumn,
    InterleavedRow,
    InterleavedDiagonal,
    InterleavedColumnComplex,
    InterleavedColumnMix,
    RepeatedRow,
    RepeatedColumn,
};

inline const char *packing_to_string(PackingType packing)
{
    switch (packing)
    {
    case PackingType::InterleavedColumn:
        return "InterleavedColumn";
    case PackingType::InterleavedRow:
        return "InterleavedRow";
    case PackingType::InterleavedDiagonal:
        return "InterleavedDiagonal";
    case PackingType::InterleavedColumnComplex:
        return "InterleavedColumnComplex";
    case PackingType::InterleavedColumnMix:
        return "InterleavedColumnMix";
    case PackingType::RepeatedRow:
        return "RepeatedRow";
    case PackingType::RepeatedColumn:
        return "RepeatedColumn";
    default:
        return "Unknown";
    }
}

inline std::string shape_to_string(const std::vector<int> &shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i != 0)
        {
            out += ",";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

struct FHEBlockTensor {
    std::vector<PhantomCiphertext> cts;
    std::string name;
    std::vector<int> logical_shape;
    PackingType packing = PackingType::Unknown;
    int level = -1;
    double scale = 0.0;
    int num_heads = 0;
    int head_dim = 0;
    bool is_complex = false;
    bool is_cached = false;
    std::string why_cached;
    std::string consumer;
    std::string backward_consumer;

    bool empty() const noexcept { return cts.empty(); }
};

struct FHEHostTensorPayload {
    std::vector<std::vector<std::uint64_t>> ciphertext_words;
    std::vector<cudaStream_t> streams;
    bool offloaded = false;
};

struct FHECacheRecord {
    FHEBlockTensor tensor;
    std::string why_cached;
    std::string consumer;
    FHEHostTensorPayload host_payload;
};

struct FHEForwardCache {
    std::unordered_map<std::string, FHECacheRecord> records;

    std::optional<FHEBlockTensor> X_input;
    std::optional<FHEBlockTensor> X_after_attn_residual;
    std::optional<FHEBlockTensor> X_after_ln1;
    std::optional<FHEBlockTensor> X_after_ffn_residual;
    std::optional<FHEBlockTensor> X_after_ln2;

    std::optional<FHEBlockTensor> XW_Q;
    std::optional<FHEBlockTensor> XW_K;
    std::optional<FHEBlockTensor> XW_V;
    std::optional<FHEBlockTensor> XA_Q;
    std::optional<FHEBlockTensor> XA_K;
    std::optional<FHEBlockTensor> XA_V;
    std::optional<FHEBlockTensor> LoRA_Q;
    std::optional<FHEBlockTensor> LoRA_K;
    std::optional<FHEBlockTensor> LoRA_V;
    std::optional<FHEBlockTensor> Q_complex;
    std::optional<FHEBlockTensor> K_complex;
    std::optional<FHEBlockTensor> V_complex;

    std::vector<FHEBlockTensor> Q_heads;
    std::vector<FHEBlockTensor> K_heads;
    std::vector<FHEBlockTensor> V_heads;
    std::vector<FHEBlockTensor> ext_QK_heads;
    std::vector<FHEBlockTensor> exp_kernel_heads;
    std::vector<FHEBlockTensor> H_heads;
    std::optional<FHEBlockTensor> H_merged;
    std::optional<FHEBlockTensor> attention_out_complex;
    std::optional<FHEBlockTensor> attention_out_real;

    std::vector<FHEBlockTensor> rot_K_heads;
    std::vector<FHEBlockTensor> real_Q_heads;
    std::vector<FHEBlockTensor> ext_U_heads;

    std::optional<FHEBlockTensor> ln1_no_norm;
    std::optional<FHEBlockTensor> ln1_inv_sqrt_var;
    std::optional<FHEBlockTensor> ln2_no_norm;
    std::optional<FHEBlockTensor> ln2_inv_sqrt_var;

    std::optional<FHEBlockTensor> fc1_complex;
    std::optional<FHEBlockTensor> fc1_real;
    std::optional<FHEBlockTensor> gelu_input;
    std::optional<FHEBlockTensor> gelu_derivative;
    std::optional<FHEBlockTensor> gelu_output;
    std::optional<FHEBlockTensor> fc2_complex;
    std::optional<FHEBlockTensor> fc2_real;

    std::optional<FHEBlockTensor> classifier_input_X1;
    std::optional<FHEBlockTensor> X1Wc1;
    std::optional<FHEBlockTensor> tanh_classifier;
    std::optional<FHEBlockTensor> X1Wc2;
    std::optional<FHEBlockTensor> y_pred;
    std::optional<FHEBlockTensor> y_lr;

    std::optional<FHEBlockTensor> XA_Q_cache;
    std::optional<FHEBlockTensor> XA_K_cache;
    std::optional<FHEBlockTensor> XA_V_cache;
};

struct FHEBackwardCache {
    std::unordered_map<std::string, FHECacheRecord> records;

    std::optional<FHEBlockTensor> grad_Wc2_packed;
    std::optional<FHEBlockTensor> pre;
    std::optional<FHEBlockTensor> grad_Wc1_packed;
    std::optional<FHEBlockTensor> gradient_x2;

    std::optional<FHEBlockTensor> gradient_LN2;
    std::optional<FHEBlockTensor> gradient_att;
    std::optional<FHEBlockTensor> gradient_LN1;
    std::optional<FHEBlockTensor> gradient_cat;
    std::optional<FHEBlockTensor> gradient_U;
    std::optional<FHEBlockTensor> D;
    std::vector<FHEBlockTensor> gradient_Q_heads;
    std::vector<FHEBlockTensor> gradient_K_heads;
    std::vector<FHEBlockTensor> gradient_V_heads;

    std::optional<FHEBlockTensor> gradient_AQ;
    std::optional<FHEBlockTensor> gradient_AK;
    std::optional<FHEBlockTensor> gradient_AV;
    std::optional<FHEBlockTensor> gradient_BQ;
    std::optional<FHEBlockTensor> gradient_BK;
    std::optional<FHEBlockTensor> gradient_BV;
    std::optional<FHEBlockTensor> gradient_X;
};

}
