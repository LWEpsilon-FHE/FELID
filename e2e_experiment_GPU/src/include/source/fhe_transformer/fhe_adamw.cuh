#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

struct AdamWHEConfig;

using AdamWHEInvSqrtBackend = std::function<FHEBlockTensor(FHEAdapterContext &,
                                                           FHEBlockTensor,
                                                           const AdamWHEConfig &,
                                                           const std::string &)>;

struct AdamWHEConfig {
    int input_level = 12;
    int update_level = 5;
    int inv_sqrt_output_level = 8;
    int timestep = 1;
    double learning_rate = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-4;
    double weight_decay = 0.0;
    double max_dividing_factor = 1.0;
    bool prescale_learning_rate_before_inv_sqrt_mul = false;
    bool keep_intermediate_tensors = true;
    bool keep_moment_tensors = false;
    bool trim_cuda_pool = false;
    PhantomBootstrappingCallbacks callbacks;
    CipherUnaryCallback gradient_bts;
    AdamWHEInvSqrtBackend inv_sqrt_backend;
};

struct AdamWHEResult {
    FHEBlockTensor updated_parameter;
    FHEBlockTensor m;
    FHEBlockTensor v;
    FHEBlockTensor m_hat;
    FHEBlockTensor v_hat;
    FHEBlockTensor v_hat_plus_epsilon;
    FHEBlockTensor inv_sqrt_v_hat_eps;
    FHEBlockTensor update_direction;
};

inline AdamWHEResult adamw_he_update_batched(FHEAdapterContext &adapter,
    const FHEBlockTensor &parameter, const std::optional<FHEBlockTensor> &previous_m,
    const std::optional<FHEBlockTensor> &previous_v, const FHEBlockTensor &gradient,
    const AdamWHEConfig &config, const std::string &name, size_t chunk_size);

inline std::string adamw_cuda_error_message(cudaError_t err, const std::string &action)
{
    return action + " failed: " + cudaGetErrorString(err);
}

inline void adamw_trim_cuda_pool_to_zero(const std::string &reason)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(adamw_cuda_error_message(err, "cudaDeviceSynchronize before mempool trim"));
    }

    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(adamw_cuda_error_message(err, "cudaGetDevice before mempool trim"));
    }

    cudaMemPool_t pool;
    err = cudaDeviceGetDefaultMemPool(&pool, device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(adamw_cuda_error_message(err, "cudaDeviceGetDefaultMemPool"));
    }

    err = cudaMemPoolTrimTo(pool, 0);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(adamw_cuda_error_message(err, "cudaMemPoolTrimTo(0)"));
    }

}

inline void adamw_trim_cuda_pool_if_enabled(const AdamWHEConfig &config,
                                            const std::string &reason)
{
    if (config.trim_cuda_pool)
    {
        adamw_trim_cuda_pool_to_zero(reason);
    }
}

inline void adamw_release_tensor_storage(FHEBlockTensor &tensor,
                                         const std::string &reason)
{
    if (!tensor.cts.empty())
    {
        
        tensor.cts.clear();
    }
    tensor.level = -1;
    tensor.scale = 0.0;
}

inline void adamw_release_intermediate_if_allowed(FHEBlockTensor &tensor,
                                                  const AdamWHEConfig &config,
                                                  const std::string &reason)
{
    if (!config.keep_intermediate_tensors)
    {
        adamw_release_tensor_storage(tensor, reason);
        adamw_trim_cuda_pool_if_enabled(config, reason);
    }
}

inline void adamw_validate_parameter_gradient_layout(const FHEBlockTensor &parameter,
                                                     const FHEBlockTensor &gradient,
                                                     const std::string &name)
{
    if (parameter.cts.size() != gradient.cts.size())
    {
        throw std::runtime_error("AdamW_HE_" + name + " ciphertext count mismatch: parameter=" +
                                 std::to_string(parameter.cts.size()) + " gradient=" +
                                 std::to_string(gradient.cts.size()));
    }
    if (!parameter.logical_shape.empty() && !gradient.logical_shape.empty() &&
        parameter.logical_shape != gradient.logical_shape)
    {
        throw std::runtime_error("AdamW_HE_" + name + " logical shape mismatch: parameter=" +
                                 shape_to_string(parameter.logical_shape) + " gradient=" +
                                 shape_to_string(gradient.logical_shape));
    }
    if (parameter.packing != PackingType::Unknown && gradient.packing != PackingType::Unknown &&
        parameter.packing != gradient.packing)
    {
        throw std::runtime_error("AdamW_HE_" + name + " packing mismatch: parameter=" +
                                 std::string(packing_to_string(parameter.packing)) +
                                 " gradient=" + packing_to_string(gradient.packing));
    }
    if (parameter.is_complex != gradient.is_complex)
    {
        throw std::runtime_error("AdamW_HE_" + name + " complex-layout mismatch");
    }
}

inline FHEBlockTensor adamw_refresh_to_input_level(FHEAdapterContext &adapter,
                                                   const FHEBlockTensor &input,
                                                   const AdamWHEConfig &config,
                                                   const std::string &name)
{
    FHEBlockTensor tensor = input;
    tensor.name = name;
    if (tensor.level > config.input_level)
    {
        align_down_to(adapter.require_context(), adapter.level_map, tensor, config.input_level, tensor.name);
        return tensor;
    }
    if (tensor.level == config.input_level)
    {
        
        return tensor;
    }
    if (!config.callbacks.ordinary_bts)
    {
        
        throw std::runtime_error("AdamW_HE " + name + " missing ordinary BTS callback");
    }
    
    if (tensor.is_complex)
        return ordinary_bts(adapter, tensor, config.callbacks, name, config.input_level,
                            "AdamW_HE_" + name + "_complex_refresh");
    align_down_to(adapter.require_context(), adapter.level_map, tensor, 0, tensor.name);
    return ordinary_bts(adapter,
                        tensor,
                        config.callbacks,
                        name,
                        config.input_level,
                        "AdamW_HE_" + name + "_ordinary_BTS_refresh");
}

inline FHEBlockTensor adamw_scalar_mul(FHEAdapterContext &adapter,
                                       const FHEBlockTensor &input,
                                       double scalar,
                                       const std::string &out_name,
                                       const std::string &step_name)
{

    FHEBlockTensor output = make_like_metadata(input, out_name);
    output.cts.resize(input.cts.size());
    
    for (size_t i = 0; i < input.cts.size(); ++i)
    {
        PhantomPlaintext plain = encode_real_constant_like(adapter, input.cts[i], scalar);
        output.cts[i] = multiply_plain_and_rescale(adapter,
                                                   input.cts[i],
                                                   plain,
                                                   step_name + "/ct_" + std::to_string(i));
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor adamw_add_plain_scalar(FHEAdapterContext &adapter,
                                             const FHEBlockTensor &input,
                                             double scalar,
                                             const std::string &out_name,
                                             const std::string &step_name)
{

    FHEBlockTensor output = input;
    output.name = out_name;
    
    for (size_t i = 0; i < output.cts.size(); ++i)
    {
        PhantomPlaintext plain = encode_real_constant_like(adapter, output.cts[i], scalar);
        adapter.require_evaluator().add_plain_inplace(output.cts[i], plain);
        
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor adamw_moment_update(FHEAdapterContext &adapter,
                                          const std::optional<FHEBlockTensor> &previous,
                                          const FHEBlockTensor &new_term,
                                          double previous_coeff,
                                          double new_coeff,
                                          const std::string &out_name,
                                          const std::string &step_name)
{
    if (previous)
    {
        FHEBlockTensor prev_scaled = adamw_scalar_mul(adapter,
                                                      *previous,
                                                      previous_coeff,
                                                      out_name + "_prev_scaled",
                                                      step_name + "_prev");
        FHEBlockTensor new_scaled = adamw_scalar_mul(adapter,
                                                     new_term,
                                                     new_coeff,
                                                     out_name + "_new_scaled",
                                                     step_name + "_new");
        FHEBlockTensor out = fhe_add(adapter,
                                     prev_scaled,
                                     new_scaled,
                                     out_name,
                                     step_name + "_add");
        return out;
    }
    return adamw_scalar_mul(adapter,
                            new_term,
                            new_coeff,
                            out_name,
                            step_name + "_init_from_gradient");
}

inline FHEBlockTensor adamw_subtract_with_scale_align(FHEAdapterContext &adapter,
                                                      const FHEBlockTensor &a,
                                                      const FHEBlockTensor &b,
                                                      const std::string &out_name,
                                                      const std::string &step_name)
{
    
    require_same_ct_count(a, b, step_name);

    FHEBlockTensor out = make_like_metadata(a, out_name);
    out.cts.resize(a.cts.size());
    for (size_t i = 0; i < a.cts.size(); ++i)
    {
        PhantomCiphertext lhs = a.cts[i];
        PhantomCiphertext rhs = b.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, lhs, rhs, a.name + "/ct_" + std::to_string(i), b.name + "/ct_" + std::to_string(i));
        align_scale_for_add(rhs, lhs,
                            b.name + "/ct_" + std::to_string(i),
                            a.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().sub(lhs, rhs, out.cts[i]);
        
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor adamw_inv_sqrt_v_hat_eps(FHEAdapterContext &adapter,
                                               FHEBlockTensor v_hat_plus_epsilon,
                                               const AdamWHEConfig &config,
                                               const std::string &name)
{
    if (!config.inv_sqrt_backend)
    {
        throw std::runtime_error("AdamW_HE_" + name +
                                 " inv_sqrt backend is not configured for 1/sqrt(v_hat+epsilon)");
    }

    FHEBlockTensor inv_input = std::move(v_hat_plus_epsilon);
    if (config.max_dividing_factor <= 0.0)
    {
        throw std::runtime_error("AdamW_HE_" + name + " max_dividing_factor must be positive");
    }
    if (config.max_dividing_factor != 1.0)
    {
        inv_input = adamw_scalar_mul(adapter,
                                     inv_input,
                                     1.0 / config.max_dividing_factor,
                                     name + "_v_hat_eps_div_max",
                                     "AdamW_HE_" + name + "_max_dividing");
    }

    AdamWHEConfig backend_config = config;
    if (config.max_dividing_factor != 1.0)
    {

        backend_config.inv_sqrt_output_level = config.inv_sqrt_output_level + 1;
    }
    FHEBlockTensor inv = config.inv_sqrt_backend(adapter,
                                                 std::move(inv_input),
                                                 backend_config,
                                                 "AdamW_HE_" + name + "_inv_sqrt_backend");
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, inv);
    if (config.max_dividing_factor != 1.0)
    {
        inv = adamw_scalar_mul(adapter,
                               inv,
                               1.0 / std::sqrt(config.max_dividing_factor),
                               name + "_inv_sqrt_v_hat_eps",
                               "AdamW_HE_" + name + "_undo_max_dividing_scale");
    }
    inv.name = name + "_inv_sqrt_v_hat_eps";
    if (inv.level > config.inv_sqrt_output_level)
    {
        align_down_to(adapter.require_context(), adapter.level_map, inv, config.inv_sqrt_output_level, inv.name);
    }
    
    return inv;
}

inline AdamWHEResult adamw_he_update(FHEAdapterContext &adapter,
                                     const FHEBlockTensor &parameter,
                                     const std::optional<FHEBlockTensor> &previous_m,
                                     const std::optional<FHEBlockTensor> &previous_v,
                                     const FHEBlockTensor &gradient,
                                     const AdamWHEConfig &config,
                                     const std::string &name)
{
    
    if (config.timestep <= 0)
    {
        throw std::runtime_error("AdamW_HE_" + name + " timestep must be positive");
    }
    if (config.learning_rate <= 0.0 || config.beta1 < 0.0 || config.beta1 >= 1.0 ||
        config.beta2 < 0.0 || config.beta2 >= 1.0 || config.epsilon <= 0.0)
    {
        throw std::runtime_error("AdamW_HE_" + name + " invalid hyperparameters");
    }
    adamw_validate_parameter_gradient_layout(parameter, gradient, name);

    if (parameter.is_complex)
        return adamw_he_update_batched(adapter, parameter, previous_m, previous_v,
                                        gradient, config, name, 8);

    std::optional<FHEBlockTensor> theta_prev_storage;
    const FHEBlockTensor *theta_prev = &parameter;

    if (parameter.level >= config.update_level + 1)
    {
        
    }
    else
    {
        theta_prev_storage = adamw_refresh_to_input_level(adapter,
                                                          parameter,
                                                          config,
                                                          name + "_theta_prev");
        theta_prev = &*theta_prev_storage;
        adamw_trim_cuda_pool_if_enabled(config, "AdamW_HE_" + name + "_theta_refresh");
    }

    std::optional<FHEBlockTensor> grad_storage;
    const FHEBlockTensor *grad = &gradient;
    if (gradient.level == config.input_level)
    {
        
    }
    else
    {
        AdamWHEConfig refresh_config = config;
        if (config.gradient_bts)
            refresh_config.callbacks.ordinary_bts = config.gradient_bts;
        grad_storage = adamw_refresh_to_input_level(adapter,
                                                    gradient,
                                                    refresh_config,
                                                    name + "_gradient");
        grad = &*grad_storage;
        adamw_trim_cuda_pool_if_enabled(config, "AdamW_HE_" + name + "_gradient_refresh");
    }

    std::optional<FHEBlockTensor> m_prev;
    std::optional<FHEBlockTensor> v_prev;
    if (previous_m)
    {
        if (previous_m->cts.size() != parameter.cts.size())
        {
            throw std::runtime_error("AdamW_HE_" + name + " previous m ciphertext count mismatch");
        }
        m_prev = adamw_refresh_to_input_level(adapter, *previous_m, config, name + "_m_prev");
    }
    if (previous_v)
    {
        if (previous_v->cts.size() != parameter.cts.size())
        {
            throw std::runtime_error("AdamW_HE_" + name + " previous v ciphertext count mismatch");
        }
        v_prev = adamw_refresh_to_input_level(adapter, *previous_v, config, name + "_v_prev");
    }

    AdamWHEResult result;
    FHEBlockTensor decayed_theta = adamw_scalar_mul(adapter,
                                                    *theta_prev,
                                                    1.0 - config.learning_rate * config.weight_decay,
                                                    name + "_theta_after_weight_decay",
                                                    "AdamW_HE_" + name + "_step3_weight_decay");
    if (theta_prev_storage)
    {
        adamw_release_intermediate_if_allowed(*theta_prev_storage,
                                              config,
                                              "AdamW_HE_" + name + "_theta_prev_storage");
    }

    const double m_bias = 1.0 - std::pow(config.beta1, config.timestep);
    const double v_bias = 1.0 - std::pow(config.beta2, config.timestep);
    FHEBlockTensor m_hat;
    
    FHEBlockTensor m = adamw_moment_update(adapter,
                                           m_prev,
                                           *grad,
                                           config.beta1,
                                           1.0 - config.beta1,
                                           name + "_m",
                                           "AdamW_HE_" + name + "_step4_m");

    FHEBlockTensor grad_square = fhe_hadamard(adapter,
                                             *grad,
                                             *grad,
                                             name + "_g_square",
                                             "AdamW_HE_" + name + "_step5_g_square");
    if (grad_storage)
    {
        adamw_release_intermediate_if_allowed(*grad_storage,
                                              config,
                                              "AdamW_HE_" + name + "_gradient_storage");
    }

    FHEBlockTensor v = adamw_moment_update(adapter,
                                           v_prev,
                                           grad_square,
                                           config.beta2,
                                           1.0 - config.beta2,
                                           name + "_v",
                                           "AdamW_HE_" + name + "_step5_v");
    FHEBlockTensor v_hat;
    
    adamw_release_intermediate_if_allowed(grad_square,
                                          config,
                                          "AdamW_HE_" + name + "_g_square");

    m_hat = adamw_scalar_mul(adapter,
                                            m,
                                            1.0 / m_bias,
                                            name + "_m_hat",
                                            "AdamW_HE_" + name + "_step6_m_hat");
    if (!config.keep_moment_tensors)
    {
        adamw_release_intermediate_if_allowed(m,
                                              config,
                                              "AdamW_HE_" + name + "_m");
    }

    v_hat = adamw_scalar_mul(adapter,
                                            v,
                                            1.0 / v_bias,
                                            name + "_v_hat",
                                            "AdamW_HE_" + name + "_step7_v_hat");
    if (!config.keep_moment_tensors)
    {
        adamw_release_intermediate_if_allowed(v,
                                              config,
                                              "AdamW_HE_" + name + "_v");
    }

    FHEBlockTensor v_hat_plus_epsilon = adamw_add_plain_scalar(adapter,
                                                               v_hat,
                                                               config.epsilon,
                                                               name + "_v_hat_plus_epsilon",
                                                               "AdamW_HE_" + name + "_step8_v_hat_plus_epsilon");
    adamw_release_intermediate_if_allowed(v_hat,
                                          config,
                                          "AdamW_HE_" + name + "_v_hat");

    FHEBlockTensor inv_sqrt_v_hat_eps =
        config.keep_intermediate_tensors
            ? adamw_inv_sqrt_v_hat_eps(adapter,
                                       v_hat_plus_epsilon,
                                       config,
                                       name)
            : adamw_inv_sqrt_v_hat_eps(adapter,
                                       std::move(v_hat_plus_epsilon),
                                       config,
                                       name);
    adamw_release_intermediate_if_allowed(v_hat_plus_epsilon,
                                          config,
                                          "AdamW_HE_" + name + "_v_hat_plus_epsilon");

    FHEBlockTensor update_left;
    const FHEBlockTensor *update_left_ptr = &m_hat;
    if (config.prescale_learning_rate_before_inv_sqrt_mul)
    {
        update_left = adamw_scalar_mul(adapter,
                                       m_hat,
                                       config.learning_rate,
                                       name + "_lr_scaled_m_hat",
                                       "AdamW_HE_" + name + "_step8_gamma_prescale_m_hat");
        update_left_ptr = &update_left;
    }

    FHEBlockTensor update_direction = fhe_hadamard(adapter,
                                                   *update_left_ptr,
                                                   inv_sqrt_v_hat_eps,
                                                   config.prescale_learning_rate_before_inv_sqrt_mul
                                                       ? name + "_scaled_update"
                                                       : name + "_m_hat_times_inv_sqrt",
                                                   "AdamW_HE_" + name + "_step8_m_hat_div_sqrt");
    adamw_release_intermediate_if_allowed(update_left,
                                          config,
                                          "AdamW_HE_" + name + "_lr_scaled_m_hat");
    adamw_release_intermediate_if_allowed(m_hat,
                                          config,
                                          "AdamW_HE_" + name + "_m_hat");
    adamw_release_intermediate_if_allowed(inv_sqrt_v_hat_eps,
                                          config,
                                          "AdamW_HE_" + name + "_inv_sqrt_v_hat_eps");

    FHEBlockTensor scaled_update =
        config.prescale_learning_rate_before_inv_sqrt_mul
            ? std::move(update_direction)
            : adamw_scalar_mul(adapter,
                               update_direction,
                               config.learning_rate,
                               name + "_scaled_update",
                               "AdamW_HE_" + name + "_step8_gamma_scale");
    if (!config.prescale_learning_rate_before_inv_sqrt_mul)
    {
        adamw_release_intermediate_if_allowed(update_direction,
                                              config,
                                              "AdamW_HE_" + name + "_update_direction");
    }

    result.updated_parameter = adamw_subtract_with_scale_align(adapter,
                                                               decayed_theta,
                                                               scaled_update,
                                                               name + "_updated",
                                                               "AdamW_HE_" + name + "_step8_theta_update");
    adamw_release_intermediate_if_allowed(decayed_theta,
                                          config,
                                          "AdamW_HE_" + name + "_decayed_theta");
    adamw_release_intermediate_if_allowed(scaled_update,
                                          config,
                                          "AdamW_HE_" + name + "_scaled_update");

    if (result.updated_parameter.level > config.update_level)
    {
        align_down_to(adapter.require_context(), adapter.level_map, result.updated_parameter, config.update_level, result.updated_parameter.name);
    }
    if (config.keep_intermediate_tensors)
    {
        result.m = std::move(m);
        result.v = std::move(v);
        result.m_hat = std::move(m_hat);
        result.v_hat = std::move(v_hat);
        result.v_hat_plus_epsilon = std::move(v_hat_plus_epsilon);
        result.inv_sqrt_v_hat_eps = std::move(inv_sqrt_v_hat_eps);
        result.update_direction = std::move(update_direction);
    }
    else if (config.keep_moment_tensors)
    {
        result.m = std::move(m);
        result.v = std::move(v);
    }
    
    return result;
}

inline AdamWHEResult adamw_he_update_batched(FHEAdapterContext &adapter,
                                            const FHEBlockTensor &parameter,
                                            const std::optional<FHEBlockTensor> &previous_m,
                                            const std::optional<FHEBlockTensor> &previous_v,
                                            const FHEBlockTensor &gradient,
                                            const AdamWHEConfig &config,
                                            const std::string &name,
                                            size_t chunk_size = 8)
{
    adamw_validate_parameter_gradient_layout(parameter, gradient, name);
    if (chunk_size == 0 || parameter.empty()) throw std::invalid_argument("Invalid AdamW chunk size/input");
    if (parameter.is_complex) {

        auto theta = convert_complex_to_real(adapter, parameter, name + "_real_lanes");
        auto grad = convert_complex_to_real(adapter, gradient, name + "_gradient_real_lanes");
        std::optional<FHEBlockTensor> m, v;
        if (previous_m) m = convert_complex_to_real(adapter, *previous_m, name + "_m_real_lanes");
        if (previous_v) v = convert_complex_to_real(adapter, *previous_v, name + "_v_real_lanes");
        auto real_update = adamw_he_update_batched(adapter, theta, m, v, grad, config,
                                                   name + "_independent_lanes", chunk_size);
        auto pack = [&](FHEBlockTensor source, int target_level, const std::string &label) {
            if (source.level < target_level)
                source = adamw_refresh_to_input_level(adapter, source, config, label + "_refresh");
            auto packed = pack_independent_complex_lanes(adapter, source, parameter, label);
            if (packed.level > target_level)
                align_down_to(adapter.require_context(), adapter.level_map, packed, target_level, label);
            return packed;
        };
        AdamWHEResult result;
        result.updated_parameter = pack(std::move(real_update.updated_parameter), config.update_level,
                                         name + "_updated");
        const int m_level = real_update.m.level;
        const int v_level = real_update.v.level;
        result.m = pack(std::move(real_update.m), m_level, name + "_m");
        result.v = pack(std::move(real_update.v), v_level, name + "_v");
        return result;
    }
    AdamWHEResult result;
    result.updated_parameter = make_like_metadata(parameter, name + "_updated");
    result.m = make_like_metadata(parameter, name + "_m");
    result.v = make_like_metadata(parameter, name + "_v");
    auto slice = [&](const FHEBlockTensor &source, size_t begin, size_t end) {
        if (source.cts.size() != parameter.cts.size())
            throw std::runtime_error(name + " optimizer state ciphertext-count mismatch");
        auto part = make_like_metadata(source, source.name + "_chunk");
        part.cts.assign(source.cts.begin() + begin, source.cts.begin() + end);
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, part);
        return part;
    };
    auto append = [](FHEBlockTensor &to, FHEBlockTensor &from) {
        for (auto &ct : from.cts) to.cts.push_back(std::move(ct));
    };
    for (size_t begin = 0; begin < parameter.cts.size(); begin += chunk_size) {
        const size_t end = std::min(parameter.cts.size(), begin + chunk_size);
        auto theta = slice(parameter, begin, end);
        auto grad = slice(gradient, begin, end);
        std::optional<FHEBlockTensor> m, v;
        if (previous_m) m = slice(*previous_m, begin, end);
        if (previous_v) v = slice(*previous_v, begin, end);
        AdamWHEConfig local = config;
        local.keep_intermediate_tensors = false;
        local.keep_moment_tensors = true;
        auto update = adamw_he_update(adapter, theta, m, v, grad, local,
                                       name + "_chunk_" + std::to_string(begin));
        append(result.updated_parameter, update.updated_parameter);
        append(result.m, update.m);
        append(result.v, update.v);
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.updated_parameter);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.m);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, result.v);
    return result;
}

inline AdamWHEResult adamw_he_update(FHEAdapterContext &adapter,
                                     const FHEBlockTensor &parameter,
                                     const FHEBlockTensor &gradient,
                                     const AdamWHEConfig &config,
                                     const std::string &name)
{
    return adamw_he_update(adapter,
                           parameter,
                           std::nullopt,
                           std::nullopt,
                           gradient,
                           config,
                           name);
}

}
