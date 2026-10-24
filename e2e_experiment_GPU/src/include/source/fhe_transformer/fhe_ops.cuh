#pragma once

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/phantom_adapter.cuh"

namespace fhe_transformer {

inline void require_same_ct_count(const FHEBlockTensor &a,
                                  const FHEBlockTensor &b,
                                  const std::string &op_name)
{
    if (a.cts.size() != b.cts.size())
    {
        throw std::runtime_error(op_name + " ciphertext count mismatch: " +
                                 std::to_string(a.cts.size()) + " vs " +
                                 std::to_string(b.cts.size()));
    }
}

inline FHEBlockTensor make_like_metadata(const FHEBlockTensor &source, const std::string &name)
{
    FHEBlockTensor out;
    out.name = name;
    out.logical_shape = source.logical_shape;
    out.packing = source.packing;
    out.num_heads = source.num_heads;
    out.head_dim = source.head_dim;
    out.is_complex = source.is_complex;
    return out;
}

template <typename Fn>
inline void run_bootstrap_batch_parallel(size_t work_items,const std::string &,const std::string &,Fn &&fn) {
    for(size_t i=0;i<work_items;++i)fn(i);
}

inline std::vector<std::unique_ptr<phantom::util::cuda_stream_wrapper>> &fhe_ct_stream_parallel_pool()
{
    static std::vector<std::unique_ptr<phantom::util::cuda_stream_wrapper>> pool;
    return pool;
}

inline phantom::util::cuda_stream_wrapper &fhe_ct_stream_parallel_stream(int tid, int nthreads)
{
    if (nthreads <= 1)
    {
        return *phantom::util::global_variables::default_stream;
    }
    auto &pool = fhe_ct_stream_parallel_pool();
    while (pool.size() < static_cast<size_t>(nthreads))
    {
        pool.emplace_back(std::make_unique<phantom::util::cuda_stream_wrapper>());
    }
    return *pool[static_cast<size_t>(tid)];
}

inline PhantomCiphertext fhe_deep_copy_cipher_on_stream(
    const PhantomCiphertext &src,
    const phantom::util::cuda_stream_wrapper &stream)
{
    PhantomCiphertext dst;
    dst.set_chain_index(src.chain_index());
    dst.set_poly_modulus_degree(src.poly_modulus_degree());
    dst.set_coeff_modulus_size(src.coeff_modulus_size());
    dst.set_scale(src.scale());
    dst.set_correction_factor(src.correction_factor());
    dst.set_ntt_form(src.is_ntt_form());
    dst.SetNoiseScaleDeg(src.GetNoiseScaleDeg());
    dst.reinit_like(src.size(),
                    src.coeff_modulus_size(),
                    src.poly_modulus_degree(),
                    stream.get_stream());
    const size_t count = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree();
    cudaMemcpyAsync(dst.data(),
                    src.data(),
                    count * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice,
                    stream.get_stream());
    return dst;
}

inline FHEBlockTensor fhe_add(FHEAdapterContext &adapter,
                              const FHEBlockTensor &a,
                              const FHEBlockTensor &b,
                              const std::string &out_name,
                              const std::string &step_name = "fhe_add")
{
    
    require_same_ct_count(a, b, "fhe_add");

    FHEBlockTensor out = make_like_metadata(a, out_name);
    out.cts.resize(a.cts.size());
    for (size_t i = 0; i < a.cts.size(); ++i)
    {
        PhantomCiphertext lhs = a.cts[i];
        PhantomCiphertext rhs = b.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, lhs, rhs, a.name + "/ct_" + std::to_string(i), b.name + "/ct_" + std::to_string(i));
        assert_scale_close(lhs, rhs, a.name, b.name);
        adapter.require_evaluator().add(lhs, rhs, out.cts[i]);
        
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor fhe_subtract(FHEAdapterContext &adapter,
                                   const FHEBlockTensor &a,
                                   const FHEBlockTensor &b,
                                   const std::string &out_name,
                                   const std::string &step_name = "fhe_subtract")
{
    
    require_same_ct_count(a, b, "fhe_subtract");

    FHEBlockTensor out = make_like_metadata(a, out_name);
    out.cts.resize(a.cts.size());
    for (size_t i = 0; i < a.cts.size(); ++i)
    {
        PhantomCiphertext lhs = a.cts[i];
        PhantomCiphertext rhs = b.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, lhs, rhs, a.name + "/ct_" + std::to_string(i), b.name + "/ct_" + std::to_string(i));
        assert_scale_close(lhs, rhs, a.name, b.name);
        adapter.require_evaluator().sub(lhs, rhs, out.cts[i]);
        
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor fhe_hadamard(FHEAdapterContext &adapter,
                                   const FHEBlockTensor &a,
                                   const FHEBlockTensor &b,
                                   const std::string &out_name,
                                   const std::string &step_name = "fhe_hadamard")
{
    
    require_same_ct_count(a, b, "fhe_hadamard");

    FHEBlockTensor out = make_like_metadata(a, out_name);
    out.cts.resize(a.cts.size());
    const int nthreads = 1;
    if (nthreads <= 1)
    {
        for (size_t i = 0; i < a.cts.size(); ++i)
        {
            PhantomCiphertext lhs = a.cts[i];
            PhantomCiphertext rhs = b.cts[i];
            const int in_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, a.name + "/ct_" + std::to_string(i), b.name + "/ct_" + std::to_string(i));
            if (in_level <= 0)
            {
                throw std::runtime_error("fhe_hadamard cannot consume a level from logical level " +
                                         std::to_string(in_level));
            }

            adapter.require_evaluator().multiply(lhs, rhs, out.cts[i]);
            adapter.require_evaluator().relinearize_inplace(out.cts[i], adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(out.cts[i]);
            
        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
        
        return out;
    }

    FHEBlockTensor lhs_tensor = a;
    FHEBlockTensor rhs_tensor = b;
    const int start_level = std::min(lhs_tensor.level, rhs_tensor.level);
    align_down_to(adapter.require_context(), adapter.level_map, lhs_tensor, start_level, lhs_tensor.name);
    align_down_to(adapter.require_context(), adapter.level_map, rhs_tensor, start_level, rhs_tensor.name);
    for (size_t i = 0; i < lhs_tensor.cts.size(); ++i)
    {
        const int in_level = get_level(adapter.level_map, lhs_tensor.cts[i]);
        if (in_level <= 0)
        {
            throw std::runtime_error("fhe_hadamard cannot consume a level from logical level " +
                                     std::to_string(in_level));
        }
         
    }

    cudaDeviceSynchronize();
    
    PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
    {
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
        const int tid = omp_get_thread_num();
        auto &stream = fhe_ct_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
        for (int idx = 0; idx < static_cast<int>(lhs_tensor.cts.size()); ++idx)
        {
            PhantomCiphertext lhs = fhe_deep_copy_cipher_on_stream(lhs_tensor.cts[static_cast<size_t>(idx)], stream);
            PhantomCiphertext rhs = fhe_deep_copy_cipher_on_stream(rhs_tensor.cts[static_cast<size_t>(idx)], stream);
            PhantomCiphertext product;
            evaluator_local.multiply(lhs, rhs, product, stream);
            evaluator_local.relinearize_inplace(product,
                                                adapter.require_relin_keys(),
                                                stream);
            evaluator_local.rescale_to_next_inplace(product, stream);
            out.cts[static_cast<size_t>(idx)] = std::move(product);
        }
        cudaStreamSynchronize(stream.get_stream());
    }
    
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor back_class_2_batch(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &y_lr,
                                         const FHEBlockTensor &tanh_classifier,
                                         const std::string &out_name = "grad_Wc2_packed")
{
    FHEBlockTensor out = fhe_hadamard(adapter, y_lr, tanh_classifier, out_name, "back_class_2_batch");
    out.packing = tanh_classifier.packing;
    out.logical_shape = tanh_classifier.logical_shape;
    out.is_complex = false;
    
    return out;
}

inline PhantomPlaintext encode_real_constant_like(FHEAdapterContext &adapter,
                                                  const PhantomCiphertext &ct,
                                                  double value)
{
    std::vector<double> values(adapter.require_ckks_encoder().slot_count(), value);
    PhantomPlaintext plain;
    encode_values(adapter, values, ct.params_id(), ct.scale(), plain);
    return plain;
}

inline PhantomPlaintext encode_complex_constant_like(FHEAdapterContext &adapter,
                                                     const PhantomCiphertext &ct,
                                                     std::complex<double> value)
{
    std::vector<std::complex<double>> values(adapter.require_ckks_encoder().slot_count(), value);
    PhantomPlaintext plain;
    encode_values(adapter, values, ct.params_id(), ct.scale(), plain);
    return plain;
}

inline PhantomCiphertext multiply_plain_and_rescale(FHEAdapterContext &adapter,
                                                    const PhantomCiphertext &ct,
                                                    PhantomPlaintext &plain,
                                                    const std::string &name)
{
    PhantomCiphertext out;
    const int in_level = get_level(adapter.level_map, ct);
    if (in_level <= 0)
    {
        throw std::runtime_error(name + " cannot consume a level from logical level " + std::to_string(in_level));
    }
    adapter.require_evaluator().multiply_plain(const_cast<PhantomCiphertext &>(ct), plain, out);
    adapter.require_evaluator().rescale_to_next_inplace(out);
    
    return out;
}

inline FHEBlockTensor convert_complex_to_real(FHEAdapterContext &adapter,
                                              const FHEBlockTensor &input,
                                              const std::string &out_name,
                                              double divisor = 1.0,
                                              const std::string &step_name = "convert_complex_to_real")
{
    
    if (!input.is_complex)
    {
        throw std::runtime_error("convert_complex_to_real requires complex input tensor " + input.name);
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = input.logical_shape;
    out.packing = PackingType::InterleavedColumn;
    out.level = input.level - 1;
    out.num_heads = input.num_heads;
    out.head_dim = input.head_dim;
    out.is_complex = false;
    out.cts.resize(input.cts.size() * 2);

    const double real_coeff = 0.5 / divisor;
    const std::complex<double> imag_coeff(0.0, -0.5 / divisor);
    for (size_t i = 0; i < input.cts.size(); ++i)
    {
        PhantomPlaintext real_plain = encode_real_constant_like(adapter, input.cts[i], real_coeff);
        PhantomPlaintext imag_plain = encode_complex_constant_like(adapter, input.cts[i], imag_coeff);

        PhantomCiphertext real_part = multiply_plain_and_rescale(adapter, input.cts[i], real_plain,
                                                                 input.name + "/real_part");
        PhantomCiphertext real_conj;
        adapter.require_evaluator().complex_conjugate(real_part, adapter.require_galois_keys(), real_conj);
        adapter.require_evaluator().add_inplace(real_part, real_conj);

        PhantomCiphertext imag_part = multiply_plain_and_rescale(adapter, input.cts[i], imag_plain,
                                                                 input.name + "/imag_part");
        PhantomCiphertext imag_conj;
        adapter.require_evaluator().complex_conjugate(imag_part, adapter.require_galois_keys(), imag_conj);
        adapter.require_evaluator().add_inplace(imag_part, imag_conj);

        out.cts[2 * i] = std::move(real_part);
        out.cts[2 * i + 1] = std::move(imag_part);
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor pack_independent_complex_lanes(FHEAdapterContext &adapter,
                                                      const FHEBlockTensor &real_lanes,
                                                      const FHEBlockTensor &layout,
                                                      const std::string &name)
{
    if (real_lanes.is_complex || real_lanes.cts.size() != 2 * layout.cts.size())
        throw std::invalid_argument("Complex lane packing count/layout mismatch");
    auto out = make_like_metadata(layout, name);
    out.cts.resize(layout.cts.size());
    for (size_t i = 0; i < out.cts.size(); ++i) {
        auto real = real_lanes.cts[2*i];
        auto imag = real_lanes.cts[2*i+1];

        PhantomPlaintext unit_i;
        encode_values(adapter, std::complex<double>(0,1), imag.params_id(), 1.0, unit_i);
        adapter.require_evaluator().multiply_plain_inplace(imag, unit_i);
        align_pair_for_add(adapter.require_context(), adapter.level_map, real, imag, name + "_real", name + "_imag");
        assert_scale_close(real, imag, name + "_real", name + "_imag");
        adapter.require_evaluator().add(real, imag, out.cts[i]);
        
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    return out;
}

inline FHEBlockTensor ordinary_bts(FHEAdapterContext &adapter,
                                   const FHEBlockTensor &input,
                                   const PhantomBootstrappingCallbacks &callbacks,
                                   const std::string &out_name,
                                   int stored_level,
                                   const std::string &step_name)
{
    if (input.is_complex) {
        if (input.level < 1)
            throw std::runtime_error(step_name + " complex refresh requires one level to split lanes");
        auto parts = convert_complex_to_real(adapter, input, out_name + "_real_lanes");
        align_down_to(adapter.require_context(), adapter.level_map, parts, 0, parts.name);
        auto refreshed = ordinary_bts(adapter, parts, callbacks, out_name + "_refreshed_lanes",
                                       stored_level, step_name + "_independent_lanes");
        return pack_independent_complex_lanes(adapter, refreshed, input, out_name);
    }
    if (!callbacks.ordinary_bts)
    {
        throw std::runtime_error("ordinary BTS callback is not configured for " + step_name);
    }
    if (input.level != 0)
    {
        std::ostringstream oss;
        oss << step_name << " ordinary BTS requires lowest logical level 0 input, got "
            << input.level << " for " << input.name
            << ". Use explicit down-align before BTS; do not treat BTS as FB.";
        
        throw std::runtime_error(oss.str());
    }

    FHEBlockTensor out = make_like_metadata(input, out_name);
    out.cts.resize(input.cts.size());
    run_bootstrap_batch_parallel(input.cts.size(),
                                 step_name,
                                 "ordinary_bts",
                                 [&](size_t i) {
                                     callbacks.ordinary_bts(out.cts[i], input.cts[i]);
                                 });
    for (size_t i = 0; i < input.cts.size(); ++i)
    {
        const int callback_level = get_level(adapter.level_map, out.cts[i]);
        if (callback_level != adapter.level_map.max_logical_level)
        {
            std::ostringstream oss;
            oss << step_name << " ordinary BTS callback returned level=" << callback_level
                << " expected=" << adapter.level_map.max_logical_level;
            
            throw std::runtime_error(oss.str());
        }
        align_down_to(adapter.require_context(), adapter.level_map, out.cts[i], stored_level, out.name + "/ct_" + std::to_string(i));
    }
    
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);

    return out;
}

inline FHEBlockTensor functional_bootstrap(FHEAdapterContext &adapter,
                                           const FHEBlockTensor &input,
                                           const CipherUnaryCallback &callback,
                                           const std::string &fb_kind,
                                           const std::string &out_name,
                                           int expected_input_level,
                                           int stored_level,
                                           const std::string &step_name)
{
    if (!callback)
    {
        throw std::runtime_error(fb_kind + " callback is not configured for " + step_name);
    }
    if (input.level != expected_input_level)
    {
        std::ostringstream oss;
        oss << fb_kind << " requires input level " << expected_input_level
            << ", got " << input.level << " for " << input.name
            << ". FB entry must reserve exactly the designed levels.";
        
        throw std::runtime_error(oss.str());
    }

    FHEBlockTensor out = make_like_metadata(input, out_name);
    out.cts.resize(input.cts.size());
    run_bootstrap_batch_parallel(input.cts.size(),
                                 step_name,
                                 fb_kind,
                                 [&](size_t i) {
                                     callback(out.cts[i], input.cts[i]);
                                 });
    for (size_t i = 0; i < input.cts.size(); ++i)
    {
        const int callback_level = get_level(adapter.level_map, out.cts[i]);
        if (callback_level != adapter.level_map.max_logical_level)
        {
            std::ostringstream oss;
            oss << fb_kind << " callback returned level=" << callback_level
                << " expected=" << adapter.level_map.max_logical_level
                << " before explicit down-align";
            
            throw std::runtime_error(oss.str());
        }
        align_down_to(adapter.require_context(), adapter.level_map, out.cts[i], stored_level, out.name + "/ct_" + std::to_string(i));
    }
    
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);

    return out;
}

inline FHEBlockTensor rboot_fb(FHEAdapterContext &adapter,
                               const FHEBlockTensor &input,
                               const PhantomBootstrappingCallbacks &callbacks,
                               const std::string &function_name,
                               const std::string &out_name,
                               int expected_input_level,
                               int stored_level,
                               const std::string &step_name)
{
    const CipherUnaryCallback *callback = nullptr;
    if (function_name == "tanh" || function_name == "gelu_tanh")
    {
        callback = &callbacks.rboot_tanh;
    }
    else if (function_name == "relu")
    {
        callback = &callbacks.rboot_relu;
    }
    else
    {
        throw std::runtime_error("unsupported RBOOT function_name=" + function_name +
                                 ". Current FB types are RBOOT(tanh/ReLU/GELU tanh) and EXP only.");
    }
    const bool polynomial=(function_name=="tanh" || function_name=="gelu_tanh") &&
                           false;
    return functional_bootstrap(adapter, input, *callback,
                                (polynomial ? "POLYNOMIAL " : "RBOOT ") + function_name,
                                out_name, expected_input_level, stored_level, step_name);
}

inline FHEBlockTensor exp_fb(FHEAdapterContext &adapter,
                             const FHEBlockTensor &input,
                             const PhantomBootstrappingCallbacks &callbacks,
                             const std::string &out_name,
                             int expected_input_level,
                             int stored_level,
                             const std::string &step_name)
{
    if (!callbacks.exp_fb_pair)
    {
        FHEBlockTensor out = functional_bootstrap(adapter, input, callbacks.exp_fb,
                                                  "EXP",
                                                  out_name, expected_input_level, stored_level, step_name);
        out.packing = PackingType::InterleavedDiagonal;
        return out;
    }
    if (input.level != expected_input_level)
    {
        std::ostringstream oss;
        oss << "EXP requires input level " << expected_input_level
            << ", got " << input.level << " for " << input.name
            << ". FB entry must reserve exactly the designed levels.";
        
        throw std::runtime_error(oss.str());
    }

    FHEBlockTensor out = make_like_metadata(input, out_name);
    out.cts.resize(input.cts.size());
    const size_t pair_count = input.cts.size() / 2;
    run_bootstrap_batch_parallel(pair_count,
                                 step_name,
                                 "EXP_pair",
                                 [&](size_t pair_index) {
                                     const size_t i = 2 * pair_index;
                                     callbacks.exp_fb_pair(out.cts[i],
                                                           out.cts[i + 1],
                                                           input.cts[i],
                                                           input.cts[i + 1]);
                                 });
    size_t i = 0;
    for (; i + 1 < input.cts.size(); i += 2)
    {
        for (size_t j = i; j < i + 2; ++j)
        {
            const int callback_level = get_level(adapter.level_map, out.cts[j]);
            if (callback_level != adapter.level_map.max_logical_level)
            {
                std::ostringstream oss;
                oss << "EXP paired callback returned level=" << callback_level
                    << " expected=" << adapter.level_map.max_logical_level
                    << " before explicit down-align";
                
                throw std::runtime_error(oss.str());
            }
            align_down_to(adapter.require_context(), adapter.level_map, out.cts[j], stored_level, out.name + "/ct_" + std::to_string(j));
        }
    }
    if (i < input.cts.size())
    {
        if (!callbacks.exp_fb)
        {
            throw std::runtime_error("EXP unary callback is not configured for odd tail in " + step_name);
        }
        callbacks.exp_fb(out.cts[i], input.cts[i]);
        const int callback_level = get_level(adapter.level_map, out.cts[i]);
        if (callback_level != adapter.level_map.max_logical_level)
        {
            std::ostringstream oss;
            oss << "EXP callback returned level=" << callback_level
                << " expected=" << adapter.level_map.max_logical_level
                << " before explicit down-align";
            
            throw std::runtime_error(oss.str());
        }
        align_down_to(adapter.require_context(), adapter.level_map, out.cts[i], stored_level, out.name + "/ct_" + std::to_string(i));
    }
    
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    out.packing = PackingType::InterleavedDiagonal;

    return out;
}

}
