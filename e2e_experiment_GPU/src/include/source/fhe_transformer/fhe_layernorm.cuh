#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/fhe_level_utils.cuh"
#include "source/fhe_transformer/fhe_plaintext_packing.cuh"

namespace fhe_transformer {

struct LayerNormParams {
    int row = 0;
    int col = 0;
    int batch_size = 1;
    int k = 1;
    double variance_c = 1.0;
    double epsilon = 1e-5;
    double output_gain = 1.0;  
    std::vector<double> gamma;
    std::vector<double> beta;
    int expected_input_level = 6;
    int expected_output_level = 5;
    int expected_no_norm_level = -1;
    int expected_inv_sqrt_var_level = -1;
};

struct LayerNormResult {
    FHEBlockTensor no_norm;
    FHEBlockTensor inv_sqrt_var;
    FHEBlockTensor normalized_input;  
    FHEBlockTensor inverse_stddev;   
    FHEBlockTensor output;
};

using LayerNormBackend = std::function<LayerNormResult(FHEAdapterContext &,
                                                       const FHEBlockTensor &,
                                                       const LayerNormParams &,
                                                       const std::string &)>;

inline void validate_layernorm_params(const FHEBlockTensor &input,
                                      const LayerNormParams &params,
                                      const std::string &step_name)
{
    if (params.row <= 0 || params.col <= 0 || params.batch_size <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " requires positive row/col/batch/k");
    }
    if (!std::isfinite(params.variance_c) || params.variance_c <= 0.0 ||
        !std::isfinite(params.epsilon) || params.epsilon <= 0.0)
    {
        throw std::runtime_error(step_name + " variance_c must be positive");
    }
    if (input.logical_shape != std::vector<int>({params.row, params.col}))
    {
        throw std::runtime_error(step_name + " input shape mismatch: actual=" +
                                 shape_to_string(input.logical_shape) + " expected=[" +
                                 std::to_string(params.row) + "," + std::to_string(params.col) + "]");
    }
    if (!params.gamma.empty() && static_cast<int>(params.gamma.size()) != params.col)
    {
        throw std::runtime_error(step_name + " gamma size must match hidden dimension");
    }
    if (!params.beta.empty() && static_cast<int>(params.beta.size()) != params.col)
    {
        throw std::runtime_error(step_name + " beta size must match hidden dimension");
    }
}

inline LayerNormResult run_layernorm(FHEAdapterContext &adapter,
                                     const FHEBlockTensor &input,
                                     const LayerNormParams &params,
                                     const LayerNormBackend &backend,
                                     const std::string &step_name)
{
    
    validate_layernorm_params(input, params, step_name);
    
    assert_level(input, params.expected_input_level, input.name);
    if (!backend)
    {
        throw std::runtime_error(step_name + " backend is not configured. "
                                 "LayerNorm cannot be represented as an opaque level jump.");
    }

    LayerNormResult result = backend(adapter, input, params, step_name);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.no_norm);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.inv_sqrt_var);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.output);

    if (result.output.level != params.expected_output_level)
    {
        throw std::runtime_error(step_name + " output level mismatch: actual=" +
                                 std::to_string(result.output.level) + " expected=" +
                                 std::to_string(params.expected_output_level));
    }
    if (result.output.logical_shape != input.logical_shape)
    {
        throw std::runtime_error(step_name + " output shape mismatch");
    }
    if (result.output.packing != PackingType::InterleavedColumn || result.output.is_complex)
    {
        throw std::runtime_error(step_name + " output must be real InterleavedColumn packing");
    }
    const int expected_no_norm_level =
        params.expected_no_norm_level >= 0 ? params.expected_no_norm_level : params.expected_output_level;
    const int expected_inv_level =
        params.expected_inv_sqrt_var_level >= 0 ? params.expected_inv_sqrt_var_level : params.expected_output_level;
    if (result.no_norm.level != expected_no_norm_level)
    {
        throw std::runtime_error(step_name + " ln_no_norm level mismatch: actual=" +
                                 std::to_string(result.no_norm.level) + " expected=" +
                                 std::to_string(expected_no_norm_level));
    }
    if (result.inv_sqrt_var.level != expected_inv_level)
    {
        throw std::runtime_error(step_name + " inv_sqrt_var level mismatch: actual=" +
                                 std::to_string(result.inv_sqrt_var.level) + " expected=" +
                                 std::to_string(expected_inv_level));
    }

    return result;
}

}
