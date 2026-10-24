#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "source/fhe_transformer/fhe_matrix_ops.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

inline void align_scale_for_add(PhantomCiphertext &a,
                                PhantomCiphertext &b,
                                const std::string &name_a,
                                const std::string &name_b)
{
    const double old_scale = a.scale();
    if (old_scale != b.scale())
    {
        const double denom = std::max({std::abs(old_scale), std::abs(b.scale()), 1.0});
        const double rel = std::abs(old_scale - b.scale()) / denom;
        
        a.scale() = b.scale();
    }
}

struct LayerNormBackwardOpParams {
    int batch_size = 1;
    int row = 128;
    int col = 768;
    std::vector<double> gamma;

    bool input_is_normalized = true;
};

struct GradientAttentionFFNOpParams {
    MatrixOpParams wf2_t_params;
    MatrixOpParams wf1_t_params;
};

struct GradientUOpParams {
    int batch_size = 1;
    int row = 128;
    int col = 64;
};

struct GradientVOpParams {
    int batch_size = 1;
    int row = 128;
    int col = 64;
    int k = 1;
};

struct GradientQKOpParams {
    int batch_size = 1;
    int row = 128;
    int col = 64;
    int k = 1;
};

struct LoraGradientOpParams {
    int batch_size = 1;
    int row = 128;
    int col = 768;
    int rank = 2;
    int k = 1;
};

inline void validate_layernorm_backward_topology(const FHEBlockTensor &gradient_x,
                                                 const FHEBlockTensor &ln_no_norm,
                                                 const LayerNormBackwardOpParams &params,
                                                 int slot_count,
                                                 const std::string &step_name)
{
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0)
    {
        throw std::runtime_error(step_name + " requires positive batch/row/col");
    }
    if (gradient_x.cts.empty() || gradient_x.cts.size() != ln_no_norm.cts.size())
    {
        throw std::runtime_error(step_name + " requires non-empty gradient_x and LN_no_norm with same ct count");
    }
    if (params.col % static_cast<int>(gradient_x.cts.size()) != 0)
    {
        throw std::runtime_error(step_name + " hidden dimension must divide evenly across ciphertexts");
    }
    const int cols_per_ct = params.col / static_cast<int>(gradient_x.cts.size());
    if (cols_per_ct * params.row * params.batch_size > slot_count)
    {
        throw std::runtime_error(step_name + " packing topology does not fit in one ciphertext");
    }
    if (slot_count % (params.row * params.batch_size) != 0)
    {
        throw std::runtime_error(step_name + " row*batch must divide slot count");
    }
    if (!params.gamma.empty() && static_cast<int>(params.gamma.size()) != params.col)
    {
        throw std::runtime_error(step_name + " gamma size must equal hidden dimension");
    }
}

inline FHEBlockTensor phantom_gradient_layernorm(FHEAdapterContext &adapter,
                                                 const FHEBlockTensor &gradient_x,
                                                 const FHEBlockTensor &ln_no_norm,
                                                 const FHEBlockTensor &inv_sqrt_var,
                                                 const LayerNormBackwardOpParams &params,
                                                 const std::string &out_name,
                                                 const std::string &step_name = "gradient_LN")
{
    
    if (gradient_x.is_complex || ln_no_norm.is_complex || inv_sqrt_var.is_complex)
    {
        throw std::runtime_error(step_name + " requires real tensors");
    }
    if (inv_sqrt_var.cts.empty())
    {
        throw std::runtime_error(step_name + " requires non-empty inv_sqrt_var");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    validate_layernorm_backward_topology(gradient_x, ln_no_norm, params, n, step_name);

    const int ct_count = static_cast<int>(gradient_x.cts.size());
    const int cols_per_ct = params.col / ct_count;
    const std::vector<double> gamma =
        params.gamma.empty() ? std::vector<double>(static_cast<size_t>(params.col), 1.0) : params.gamma;

    std::vector<PhantomCiphertext> H(static_cast<size_t>(ct_count));
    for (int i = 0; i < ct_count; ++i)
    {
        std::vector<double> gamma_i(static_cast<size_t>(n), 0.0);
        for (int j = 0; j < cols_per_ct; ++j)
        {
            const double coeff = gamma[static_cast<size_t>(i * cols_per_ct + j)] /
                                 static_cast<double>(params.col);
            for (int k = 0; k < params.row * params.batch_size; ++k)
            {
                gamma_i[static_cast<size_t>(j * params.row * params.batch_size + k)] = coeff;
            }
        }
        PhantomPlaintext encoded_gamma;
        encode_values(adapter,
                          gamma_i,
                          gradient_x.cts[static_cast<size_t>(i)].params_id(),
                          gradient_x.cts[static_cast<size_t>(i)].scale(),
                          encoded_gamma);
        PhantomCiphertext gx = gradient_x.cts[static_cast<size_t>(i)];
        adapter.require_evaluator().multiply_plain(gx, encoded_gamma, H[static_cast<size_t>(i)]);
        adapter.require_evaluator().rescale_to_next_inplace(H[static_cast<size_t>(i)]);
        
    }

    PhantomCiphertext m = H.front();
    for (int i = 1; i < ct_count; ++i)
    {
        PhantomCiphertext h = H[static_cast<size_t>(i)];
        align_pair_for_add(adapter.require_context(), adapter.level_map, m, h, step_name + "/m", step_name + "/H_" + std::to_string(i));
        align_scale_for_add(h, m, step_name + "/H_" + std::to_string(i), step_name + "/m");
        adapter.require_evaluator().add_inplace(m, h);
        
    }

    int rot_index = n / 2;
    while (rot_index >= params.row * params.batch_size)
    {
        PhantomCiphertext tmp;
        adapter.require_evaluator().rotate_vector(m, rot_index, adapter.require_galois_keys(), tmp);
        adapter.require_evaluator().add_inplace(m, tmp);
        
        rot_index /= 2;
    }

    PhantomCiphertext v;
    bool v_set = false;
    for (int i = 0; i < ct_count; ++i)
    {
        PhantomCiphertext z = ln_no_norm.cts[static_cast<size_t>(i)];
        PhantomCiphertext h = H[static_cast<size_t>(i)];
        align_pair_for_mul(adapter.require_context(), adapter.level_map, z, h, ln_no_norm.name + "/ct_" + std::to_string(i), step_name + "/H_" + std::to_string(i));
        PhantomCiphertext product;
        adapter.require_evaluator().multiply(z, h, product);
        
        if (!v_set)
        {
            v = std::move(product);
            v_set = true;
        }
        else
        {
            align_scale_for_add(product, v, step_name + "/ZH_" + std::to_string(i), step_name + "/v");
            adapter.require_evaluator().add_inplace(v, product);
            
        }
    }
    adapter.require_evaluator().relinearize_inplace(v, adapter.require_relin_keys());
    adapter.require_evaluator().rescale_to_next_inplace(v);

    rot_index = n / 2;
    while (rot_index >= params.row * params.batch_size)
    {
        PhantomCiphertext tmp;
        adapter.require_evaluator().rotate_vector(v, rot_index, adapter.require_galois_keys(), tmp);
        adapter.require_evaluator().add_inplace(v, tmp);
        
        rot_index /= 2;
    }

    if (!params.input_is_normalized)
    {
        if (inv_sqrt_var.cts.size() != 1)
            throw std::runtime_error(step_name + " raw cache requires one row-replicated inverse stddev");
        PhantomCiphertext inv2 = inv_sqrt_var.cts.front();
        adapter.require_evaluator().square_inplace(inv2);
        adapter.require_evaluator().relinearize_inplace(inv2, adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(inv2);
        align_pair_for_mul(adapter.require_context(), adapter.level_map, v, inv2, step_name + "/projection", step_name + "/inv_stddev_squared");
        adapter.require_evaluator().multiply_inplace(v, inv2);
        adapter.require_evaluator().relinearize_inplace(v, adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(v);
        
    }

    FHEBlockTensor out;
    out.name = out_name;
    out.logical_shape = gradient_x.logical_shape;
    out.packing = gradient_x.packing;
    out.is_complex = false;
    out.cts.resize(gradient_x.cts.size());

    for (int i = 0; i < ct_count; ++i)
    {
        PhantomCiphertext z = ln_no_norm.cts[static_cast<size_t>(i)];
        PhantomCiphertext v_copy = v;
        align_pair_for_mul(adapter.require_context(), adapter.level_map, z, v_copy, ln_no_norm.name + "/ct_" + std::to_string(i), step_name + "/v");
        PhantomCiphertext zv;
        adapter.require_evaluator().multiply(z, v_copy, zv);
        adapter.require_evaluator().relinearize_inplace(zv, adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(zv);

        PhantomCiphertext centered = H[static_cast<size_t>(i)];

        PhantomPlaintext hidden_size = encode_real_constant_like(adapter, centered,
                                                                 static_cast<double>(params.col));
        adapter.require_evaluator().multiply_plain_inplace(centered, hidden_size);
        adapter.require_evaluator().rescale_to_next_inplace(centered);
        
        PhantomCiphertext m_copy = m;
        align_pair_for_add(adapter.require_context(), adapter.level_map, centered, m_copy, step_name + "/H_" + std::to_string(i), step_name + "/m");
        align_scale_for_add(m_copy, centered, step_name + "/m", step_name + "/H_" + std::to_string(i));
        adapter.require_evaluator().sub_inplace(centered, m_copy);

        align_pair_for_add(adapter.require_context(), adapter.level_map, centered, zv, step_name + "/H_minus_m_" + std::to_string(i), step_name + "/zv_" + std::to_string(i));
        align_scale_for_add(centered, zv,
                            step_name + "/H_minus_m_" + std::to_string(i),
                            step_name + "/zv_" + std::to_string(i));
        adapter.require_evaluator().sub_inplace(centered, zv);

        const PhantomCiphertext &inv_source =
            inv_sqrt_var.cts.size() == 1
                ? inv_sqrt_var.cts.front()
                : inv_sqrt_var.cts.at(static_cast<size_t>(i));
        PhantomCiphertext inv = inv_source;
        const int mul_level = align_pair_for_mul(adapter.require_context(), adapter.level_map, centered, inv, step_name + "/centered_" + std::to_string(i), inv_sqrt_var.name + "/ct_" +
                                                     std::to_string(inv_sqrt_var.cts.size() == 1 ? 0 : i));
        if (mul_level <= 0)
        {
            throw std::runtime_error(step_name + " cannot multiply by inv_sqrt_var at level " +
                                     std::to_string(mul_level));
        }
        adapter.require_evaluator().multiply(centered, inv, out.cts[static_cast<size_t>(i)]);
        adapter.require_evaluator().relinearize_inplace(out.cts[static_cast<size_t>(i)],
                                                        adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(out.cts[static_cast<size_t>(i)]);
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, out);
    
    return out;
}

inline FHEBlockTensor phantom_gradient_cat(FHEAdapterContext &adapter,
                                           const FHEBlockTensor &gradient_ln1,
                                           const std::vector<std::vector<double>> &W_o,
                                           const MatrixOpParams &params,
                                           const std::string &out_name,
                                           const std::string &step_name = "gradient_cat")
{
    
    if (gradient_ln1.is_complex)
    {
        throw std::runtime_error(step_name + " requires real gradient_LN1 input");
    }
    if (params.col_X != params.col_W || params.row_W != params.col_X)
    {
        throw std::runtime_error(step_name + " expects square Wo-compatible PCMM parameters");
    }
    if (static_cast<int>(W_o.size()) != params.col_W ||
        (W_o.empty() ? 0 : static_cast<int>(W_o.front().size())) != params.col_W)
    {
        throw std::runtime_error(step_name + " Wo plaintext matrix shape mismatch");
    }

    const std::vector<std::vector<double>> WoT = transpose_matrix(W_o);
    FHEBlockTensor output = phantom_pcmm_real(adapter,
                                             gradient_ln1,
                                             WoT,
                                             params,
                                             out_name,
                                             step_name + "_PCMM_WoT");
    output.name = out_name;
    return output;
}

inline FHEBlockTensor phantom_gradient_attention_ffn(FHEAdapterContext &adapter,
                                                     const FHEBlockTensor &gradient_ln,
                                                     const FHEBlockTensor &gelu_derivative,
                                                     const FHEBlockTensor &gelu_input,
                                                     const std::vector<std::vector<double>> &W_fc1,
                                                     const std::vector<std::vector<double>> &W_fc2,
                                                     const GradientAttentionFFNOpParams &params,
                                                     const std::string &out_name,
                                                     const std::string &step_name = "gradient_att")
{
    
    if (gradient_ln.is_complex || gelu_derivative.is_complex || gelu_input.is_complex)
    {
        throw std::runtime_error(step_name + " requires real tensors");
    }
    if (W_fc1.empty() || W_fc2.empty())
    {
        throw std::runtime_error(step_name + " requires plaintext W_fc1/W_fc2 matrices");
    }

    FHEBlockTensor hidden_grad = phantom_pcmm_real(adapter,
                                                   gradient_ln,
                                                   transpose_matrix(W_fc2),
                                                   params.wf2_t_params,
                                                   out_name + "_after_Wfc2T",
                                                   step_name + "_PCMM_Wfc2T");
    if (hidden_grad.cts.size() != gelu_derivative.cts.size() ||
        hidden_grad.cts.size() != gelu_input.cts.size())
    {
        throw std::runtime_error(step_name + " hidden gradient, gelu_derivative, and gelu_input ct counts must match");
    }

    hidden_grad=fhe_hadamard(adapter,hidden_grad,gelu_derivative,hidden_grad.name,step_name+"_gelu_derivative");
    align_down_to(adapter.require_context(),adapter.level_map,hidden_grad,1,hidden_grad.name);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, hidden_grad);

    FHEBlockTensor output = phantom_pcmm_real(adapter,
                                              hidden_grad,
                                              transpose_matrix(W_fc1),
                                              params.wf1_t_params,
                                              out_name + "_pre_residual",
                                              step_name + "_PCMM_Wfc1T");
    if (output.cts.size() != gradient_ln.cts.size())
    {
        throw std::runtime_error(step_name + " output and residual gradient ct counts must match");
    }
    for (size_t i = 0; i < output.cts.size(); ++i)
    {
        PhantomCiphertext residual = gradient_ln.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, output.cts[i], residual, output.name + "/ct_" + std::to_string(i), gradient_ln.name + "/ct_" + std::to_string(i));
        align_scale_for_add(residual, output.cts[i],
                            gradient_ln.name + "/ct_" + std::to_string(i),
                            output.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().add_inplace(output.cts[i], residual);
        
    }

    output.name = out_name;
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline std::vector<PhantomCiphertext> phantom_rot_g_bs(FHEAdapterContext &adapter,
                                                       const FHEBlockTensor &gradient_cat_head,
                                                       const GradientUOpParams &params,
                                                       const std::string &step_name)
{
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0)
    {
        throw std::runtime_error(step_name + " requires positive batch/row/col");
    }
    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = static_cast<int>(gradient_cat_head.cts.size());
    const int ed = params.row * params.row * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row * params.row * params.batch_size % n != 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        mid != params.row * params.col * params.batch_size / n ||
        params.row % ed != 0)
    {
        throw std::runtime_error(step_name + " gradient_cat topology does not match CPU rot_G_bs");
    }

    std::vector<PhantomCiphertext> rot_g(static_cast<size_t>(ed * mid));
    const int rot_work_items = ed * mid;
    const int nthreads = 1;
    if (nthreads <= 1)
    {
        for (int ind = 0; ind < ed; ++ind)
        {
            for (int j = 0; j < mid; ++j)
            {
                rot_g[static_cast<size_t>(ind * mid + j)] = gradient_cat_head.cts[static_cast<size_t>(j)];
                phantom_internal_rot_inplace(adapter,
                                             rot_g[static_cast<size_t>(ind * mid + j)],
                                             params.row - (ind * params.row / ed),
                                             params.row,
                                             params.batch_size,
                                             1,
                                             gradient_cat_head.name + "_rot_G_" +
                                                 std::to_string(ind) + "_" + std::to_string(j));
            }
        }
        return rot_g;
    }

    cudaDeviceSynchronize();
    PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
    {
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Encoder encoder_local(&context, &phantom_encoder_local);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
        const int tid = omp_get_thread_num();
        auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
        for (int work = 0; work < rot_work_items; ++work)
        {
            const int ind = work / mid;
            const int j = work % mid;
            PhantomCiphertext tmp = fhe_matrix_deep_copy_cipher_on_stream(
                gradient_cat_head.cts[static_cast<size_t>(j)],
                stream);
            phantom_internal_rot_inplace_stream(adapter,
                                                encoder_local,
                                                evaluator_local,
                                                stream,
                                                tmp,
                                                params.row - (ind * params.row / ed),
                                                params.row,
                                                params.batch_size,
                                                1);
            rot_g[static_cast<size_t>(work)] = std::move(tmp);
        }
        cudaStreamSynchronize(stream.get_stream());
    }
    
    return rot_g;
}

inline FHEBlockTensor phantom_gradient_u(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &gradient_cat_head,
                                         const FHEBlockTensor &V_head,
                                         const GradientUOpParams &params,
                                         const std::string &out_name,
                                         const std::string &step_name = "gradient_U")
{
    
    if (gradient_cat_head.is_complex || !V_head.is_complex)
    {
        throw std::runtime_error(step_name + " requires real gradient_cat head and complex V head");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = params.row * params.col * params.batch_size / n;
    const int ed = params.row * params.row * params.batch_size / n;
    if (mid <= 0 || ed <= 0 || params.row % ed != 0 ||
        static_cast<int>(gradient_cat_head.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " gradient_cat head shape/count mismatch");
    }

    std::vector<PhantomCiphertext> rot_g = phantom_rot_g_bs(adapter,
                                                            gradient_cat_head,
                                                            params,
                                                            step_name);
    FHEBlockTensor real_v = convert_complex_to_real(adapter,
                                                    V_head,
                                                    V_head.name + "_real_for_gradient_U",
                                                    -std::sqrt(static_cast<double>(params.col)),
                                                    step_name + "_Complex_to_real_back_V");
    if (static_cast<int>(real_v.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " real V ciphertext count mismatch");
    }

    const int num_diag = params.row / ed;

    std::vector<PhantomCiphertext> rot_v(static_cast<size_t>(num_diag * mid));
    const int rot_v_work_items = num_diag * mid;
    const int rot_v_threads = 1;
    if (rot_v_threads <= 1)
    {
        for (int i = 0; i < num_diag; ++i)
        {
            for (int j = 0; j < mid; ++j)
            {
                rot_v[static_cast<size_t>(i * mid + j)] = real_v.cts[static_cast<size_t>(j)];
                phantom_internal_rot_inplace(adapter,
                                             rot_v[static_cast<size_t>(i * mid + j)],
                                             i,
                                             params.row,
                                             params.batch_size,
                                             1,
                                             real_v.name + "_rot_V_" +
                                                 std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }
    else
    {
        
        cudaDeviceSynchronize();
        PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(rot_v_threads)
        {
            PhantomCKKSEncoder phantom_encoder_local(context);
            moai::Encoder encoder_local(&context, &phantom_encoder_local);
            moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
            const int tid = omp_get_thread_num();
            auto &stream = fhe_stream_parallel_stream(tid, rot_v_threads);

#pragma omp for schedule(static)
            for (int work = 0; work < rot_v_work_items; ++work)
            {
                const int i = work / mid;
                const int j = work % mid;
                PhantomCiphertext tmp = fhe_matrix_deep_copy_cipher_on_stream(
                    real_v.cts[static_cast<size_t>(j)],
                    stream);
                phantom_internal_rot_inplace_stream(adapter,
                                                    encoder_local,
                                                    evaluator_local,
                                                    stream,
                                                    tmp,
                                                    i,
                                                    params.row,
                                                    params.batch_size,
                                                    1);
                rot_v[static_cast<size_t>(work)] = std::move(tmp);
            }
            cudaStreamSynchronize(stream.get_stream());
        }
        
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.row};
    output.packing = PackingType::InterleavedDiagonal;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(params.row));

    const int output_work_items = ed * num_diag;
    const int output_threads = 1;
    if (output_threads <= 1)
    {
        for (int ind = 0; ind < ed; ++ind)
        {
            for (int i = 0; i < num_diag; ++i)
            {
                PhantomCiphertext tmpoutput;
                bool tmpoutput_set = false;
                for (int j = 0; j < mid; ++j)
                {
                    PhantomCiphertext lhs = rot_g[static_cast<size_t>(ind * mid + j)];
                    PhantomCiphertext rhs = rot_v[static_cast<size_t>(i * mid + j)];
                    align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, gradient_cat_head.name + "_rot_G_" + std::to_string(ind) + "_" + std::to_string(j), real_v.name + "_rot_V_" + std::to_string(i) + "_" + std::to_string(j));
                    PhantomCiphertext product;
                    adapter.require_evaluator().multiply(lhs, rhs, product);
                    
                    if (!tmpoutput_set)
                    {
                        tmpoutput = std::move(product);
                        tmpoutput_set = true;
                    }
                    else
                    {
                        align_scale_for_add(product, tmpoutput,
                                            step_name + "/GV_" + std::to_string(j),
                                            step_name + "/tmpoutput");
                        adapter.require_evaluator().add_inplace(tmpoutput, product);
                        
                    }
                }
                adapter.require_evaluator().relinearize_inplace(tmpoutput, adapter.require_relin_keys());
                adapter.require_evaluator().rescale_to_next_inplace(tmpoutput);

                int sum_index = n / 2;
                const int min_index = params.row * params.batch_size;
                while (sum_index >= min_index)
                {
                    PhantomCiphertext copy;
                    adapter.require_evaluator().rotate_vector(tmpoutput, sum_index, adapter.require_galois_keys(), copy);
                    align_scale_for_add(copy, tmpoutput,
                                        step_name + "/row_sum_rot_" + std::to_string(sum_index),
                                        step_name + "/tmpoutput");
                    adapter.require_evaluator().add_inplace(tmpoutput, copy);
                    
                    sum_index /= 2;
                }

                output.cts[static_cast<size_t>(ind * num_diag + i)] = std::move(tmpoutput);
            }
        }
    }
    else
    {
        
        int row_sum_rotations = 0;
        for (int sum_index = n / 2;
             sum_index >= params.row * params.batch_size;
             sum_index /= 2)
        {
            ++row_sum_rotations;
        }
        cudaDeviceSynchronize();
        PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
        long long parallel_mod_switches = 0;
#pragma omp parallel num_threads(output_threads) reduction(+ : parallel_mod_switches)
        {
            PhantomCKKSEncoder phantom_encoder_local(context);
            moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
            const int tid = omp_get_thread_num();
            auto &stream = fhe_stream_parallel_stream(tid, output_threads);

#pragma omp for schedule(static)
            for (int work = 0; work < output_work_items; ++work)
            {
                const int ind = work / num_diag;
                const int i = work % num_diag;
                PhantomCiphertext tmpoutput;
                bool tmpoutput_set = false;
                for (int j = 0; j < mid; ++j)
                {
                    PhantomCiphertext lhs = fhe_matrix_deep_copy_cipher_on_stream(
                        rot_g[static_cast<size_t>(ind * mid + j)],
                        stream);
                    PhantomCiphertext rhs = fhe_matrix_deep_copy_cipher_on_stream(
                        rot_v[static_cast<size_t>(i * mid + j)],
                        stream);
                    fhe_align_pair_for_mul_stream_quiet(context,
                                                        adapter.level_map,
                                                        lhs,
                                                        rhs,
                                                        stream,
                                                        parallel_mod_switches);
                    PhantomCiphertext product;
                    evaluator_local.multiply(lhs, rhs, product, stream);
                    if (!tmpoutput_set)
                    {
                        tmpoutput = std::move(product);
                        tmpoutput_set = true;
                    }
                    else
                    {
                        align_scale_for_add(product, tmpoutput,
                                            step_name + "/GV_stream",
                                            step_name + "/tmpoutput_stream");
                        evaluator_local.add_inplace(tmpoutput, product, stream);
                    }
                }
                evaluator_local.relinearize_inplace(tmpoutput, adapter.require_relin_keys(), stream);
                evaluator_local.rescale_to_next_inplace(tmpoutput, stream);

                int sum_index = n / 2;
                const int min_index = params.row * params.batch_size;
                while (sum_index >= min_index)
                {
                    PhantomCiphertext copy;
                    evaluator_local.rotate_vector(tmpoutput, sum_index, adapter.require_galois_keys(), copy, stream);
                    align_scale_for_add(copy, tmpoutput,
                                        step_name + "/row_sum_rot_stream",
                                        step_name + "/tmpoutput_stream");
                    evaluator_local.add_inplace(tmpoutput, copy, stream);
                    sum_index /= 2;
                }
                output.cts[static_cast<size_t>(work)] = std::move(tmpoutput);
            }
            cudaStreamSynchronize(stream.get_stream());
        }
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_dk_batch_bsgs_complex(FHEAdapterContext &adapter,
                                                         const FHEBlockTensor &ext_d,
                                                         const FHEBlockTensor &rot_k,
                                                         const GradientQKOpParams &params,
                                                         const std::string &out_name,
                                                         const std::string &step_name = "CCMM_DK_batch_bsgs_complex")
{
    
    if (ext_d.is_complex || rot_k.is_complex)
    {
        throw std::runtime_error(step_name + " requires real ext_D and rotated K/Q tensors");
    }
    if (ext_d.packing != PackingType::InterleavedDiagonal)
    {
        throw std::runtime_error(step_name + " requires ext_D in interleaved diagonal packing");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int midd = params.row * params.row * params.batch_size / n;
    const int midk = params.row * params.col * params.batch_size / n;
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0 || params.k <= 0 ||
        midd <= 0 || midk <= 0 ||
        params.row * params.row * params.batch_size % n != 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        params.row % midd != 0)
    {
        throw std::runtime_error(step_name + " topology does not match CPU CCMM_DK_batch_bsgs_complex");
    }
    const int num_diag = params.row / midd;
    if (static_cast<int>(ext_d.cts.size()) < params.row ||
        static_cast<int>(rot_k.cts.size()) != midk * num_diag)
    {
        throw std::runtime_error(step_name + " ciphertext count mismatch");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(midk));

    for (int ind = 0; ind < midk; ++ind)
    {
        bool output_set = false;
        for (int i = 0; i < midd; ++i)
        {
            PhantomCiphertext tmpdk;
            bool tmpdk_set = false;
            for (int j = 0; j < num_diag; ++j)
            {
                const int diag_index = i * num_diag + j;
                PhantomCiphertext lhs = ext_d.cts[static_cast<size_t>(diag_index)];
                PhantomCiphertext rhs = rot_k.cts[static_cast<size_t>(ind * num_diag + j)];
                align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, ext_d.name + "/diag_" + std::to_string(diag_index), rot_k.name + "/rot_" + std::to_string(ind) +
                                       "_" + std::to_string(j));
                PhantomCiphertext product;
                adapter.require_evaluator().multiply(lhs, rhs, product);
                
                if (!tmpdk_set)
                {
                    tmpdk = std::move(product);
                    tmpdk_set = true;
                }
                else
                {
                    align_scale_for_add(product, tmpdk,
                                        step_name + "/product_" + std::to_string(ind) +
                                            "_" + std::to_string(diag_index),
                                        step_name + "/tmpdk");
                    adapter.require_evaluator().add_inplace(tmpdk, product);
                    
                }
            }

            adapter.require_evaluator().relinearize_inplace(tmpdk, adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(tmpdk);

            phantom_internal_rot_inplace(adapter,
                                         tmpdk,
                                         i * params.row / midd,
                                         params.row,
                                         params.batch_size,
                                         params.k,
                                         step_name + "_tmpdk_rot_" + std::to_string(ind) +
                                             "_" + std::to_string(i));

            if (!output_set)
            {
                output.cts[static_cast<size_t>(ind)] = std::move(tmpdk);
                output_set = true;
            }
            else
            {
                align_scale_for_add(tmpdk, output.cts[static_cast<size_t>(ind)],
                                    step_name + "/tmpdk_" + std::to_string(ind) +
                                        "_" + std::to_string(i),
                                    step_name + "/output_" + std::to_string(ind));
                adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(ind)], tmpdk);
                
            }
        }
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline PhantomCiphertext phantom_sum_diag_groups(FHEAdapterContext &adapter,
                                                 const std::vector<PhantomCiphertext> &diagonals,
                                                 const std::string &source_name,
                                                 const GradientQKOpParams &params,
                                                 bool rotate_groups,
                                                 const std::string &step_name)
{
    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int midd = params.row * params.row * params.batch_size / n;
    if (midd <= 0 ||
        params.row * params.row * params.batch_size % n != 0 ||
        params.row % midd != 0 ||
        static_cast<int>(diagonals.size()) < params.row)
    {
        throw std::runtime_error(step_name + " cannot sum diagonal groups for this topology");
    }
    const int num_diag = params.row / midd;
    PhantomCiphertext summed;
    bool summed_set = false;
    for (int i = 0; i < midd; ++i)
    {
        PhantomCiphertext group = diagonals[static_cast<size_t>(i * num_diag)];
        for (int j = 1; j < num_diag; ++j)
        {
            PhantomCiphertext addend = diagonals[static_cast<size_t>(i * num_diag + j)];
            align_pair_for_add(adapter.require_context(), adapter.level_map, group, addend, source_name + "/group_" + std::to_string(i), source_name + "/diag_" + std::to_string(i * num_diag + j));
            align_scale_for_add(addend, group,
                                source_name + "/diag_" + std::to_string(i * num_diag + j),
                                source_name + "/group_" + std::to_string(i));
            adapter.require_evaluator().add_inplace(group, addend);
            
        }
        if (rotate_groups)
        {
            adapter.require_evaluator().rotate_vector_inplace(group,
                                                              num_diag * i * params.batch_size,
                                                              adapter.require_galois_keys());
            
        }
        if (!summed_set)
        {
            summed = std::move(group);
            summed_set = true;
        }
        else
        {
            align_pair_for_add(adapter.require_context(), adapter.level_map, summed, group, step_name + "/sum", source_name + "/group_" + std::to_string(i));
            align_scale_for_add(group, summed,
                                source_name + "/group_" + std::to_string(i),
                                step_name + "/sum");
            adapter.require_evaluator().add_inplace(summed, group);
            
        }
    }
    return summed;
}

inline FHEBlockTensor phantom_gradient_q(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &D,
                                         const FHEBlockTensor &Q,
                                         const FHEBlockTensor &rot_K,
                                         const GradientQKOpParams &params,
                                         const std::string &out_name,
                                         const std::string &step_name = "gradient_Q")
{
    
    if (D.is_complex || rot_K.is_complex)
    {
        throw std::runtime_error(step_name + " requires real D and rot_K tensors");
    }

    FHEBlockTensor DK = phantom_ccmm_dk_batch_bsgs_complex(adapter,
                                                           D,
                                                           rot_K,
                                                           params,
                                                           out_name + "_DK",
                                                           step_name + "_CCMM_DK");
    PhantomCiphertext sum_D = phantom_sum_diag_groups(adapter,
                                                       D.cts,
                                                       D.name,
                                                       params,
                                                       true,
                                                       step_name + "_sum_D");
    FHEBlockTensor real_Q = Q.is_complex
                                ? convert_complex_to_real(adapter,
                                                          Q,
                                                          Q.name + "_real_for_gradient_Q",
                                                          2.0,
                                                          step_name + "_Complex_to_real_back_Q")
                                : Q;
    if (real_Q.is_complex)
    {
        throw std::runtime_error(step_name + " requires Q to be real or convertible to real");
    }
    if (real_Q.cts.size() != DK.cts.size())
    {
        throw std::runtime_error(step_name + " Q and DK ciphertext counts must match");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(real_Q.cts.size());

    for (size_t i = 0; i < real_Q.cts.size(); ++i)
    {
        PhantomCiphertext lhs = sum_D;
        PhantomCiphertext rhs = real_Q.cts[i];
        align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, step_name + "/sum_D", real_Q.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().multiply(lhs, rhs, output.cts[i]);
        adapter.require_evaluator().relinearize_inplace(output.cts[i], adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(output.cts[i]);

        PhantomCiphertext dk = DK.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, output.cts[i], dk, output.name + "/ct_" + std::to_string(i), DK.name + "/ct_" + std::to_string(i));
        align_scale_for_add(dk, output.cts[i],
                            DK.name + "/ct_" + std::to_string(i),
                            output.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().sub_inplace(output.cts[i], dk);
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline std::vector<PhantomCiphertext> phantom_rot_ut_grad_v(FHEAdapterContext &adapter,
                                                            const FHEBlockTensor &ext_u,
                                                            const GradientVOpParams &params,
                                                            const std::string &step_name)
{
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " requires positive batch/row/col/k");
    }
    if (static_cast<int>(ext_u.cts.size()) < params.row)
    {
        throw std::runtime_error(step_name + " ext_U must contain one ciphertext per diagonal");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int ed = params.row * params.col * params.batch_size / n;
    if (ed <= 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        params.row % (2 * ed) != 0)
    {
        throw std::runtime_error(step_name + " ext_U topology does not match CPU rot_UT_gradV/CCMM_V_batch");
    }
    const int num_diag = params.row / (2 * ed);

    std::vector<PhantomCiphertext> ut(static_cast<size_t>(params.row));
    const int nthreads = 1;
    if (nthreads <= 1)
    {
        for (int i = 0; i < params.row; ++i)
        {
            const int source_index = (params.row - i) % params.row;
            ut[static_cast<size_t>(i)] = ext_u.cts[static_cast<size_t>(source_index)];
            const int u_idx = params.row - i;
            int left_rot_step = u_idx - (u_idx % num_diag);
            left_rot_step += i;
            while (left_rot_step < 0)
            {
                left_rot_step += params.row;
            }
            adapter.require_evaluator().rotate_vector_inplace(ut[static_cast<size_t>(i)],
                                                              left_rot_step * params.batch_size,
                                                              adapter.require_galois_keys());
            
        }
        return ut;
    }

    cudaDeviceSynchronize();
    PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
    {
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
        const int tid = omp_get_thread_num();
        auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
        for (int i = 0; i < params.row; ++i)
        {
            const int source_index = (params.row - i) % params.row;
            PhantomCiphertext tmp = fhe_matrix_deep_copy_cipher_on_stream(
                ext_u.cts[static_cast<size_t>(source_index)],
                stream);
            const int u_idx = params.row - i;
            int left_rot_step = u_idx - (u_idx % num_diag);
            left_rot_step += i;
            while (left_rot_step < 0)
            {
                left_rot_step += params.row;
            }
            evaluator_local.rotate_vector_inplace(tmp,
                                                  left_rot_step * params.batch_size,
                                                  adapter.require_galois_keys(),
                                                  stream);
            ut[static_cast<size_t>(i)] = std::move(tmp);
        }
        cudaStreamSynchronize(stream.get_stream());
    }
    
    return ut;
}

inline FHEBlockTensor phantom_ccmm_v_batch_from_ut(FHEAdapterContext &adapter,
                                                   const std::vector<PhantomCiphertext> &ut,
                                                   const FHEBlockTensor &enc_v,
                                                   const GradientVOpParams &params,
                                                   const std::string &out_name,
                                                   const std::string &step_name)
{
    if (enc_v.is_complex)
    {
        throw std::runtime_error(step_name + " requires real enc_V/gradient tensor");
    }
    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int midv = static_cast<int>(enc_v.cts.size());
    const int expected_midv = params.row * params.col * params.batch_size / n;
    if (midv <= 0 || expected_midv != midv ||
        params.row * params.col * params.batch_size % n != 0)
    {
        throw std::runtime_error(step_name + " enc_V ciphertext count mismatch");
    }
    const int midqk = midv * 2;
    if (params.row % midqk != 0 || static_cast<int>(ut.size()) < params.row)
    {
        throw std::runtime_error(step_name + " CCMM_V_batch topology mismatch");
    }
    const int num_diag = params.row / midqk;

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(midv));

    const int nthreads = 1;
    if (nthreads <= 1)
    {
        for (int ind = 0; ind < midv; ++ind)
        {
            bool output_set = false;
            for (int i = 0; i < midqk; ++i)
            {
                for (int j = 0; j < num_diag; ++j)
                {
                    const int diag_index = i * num_diag + j;
                    PhantomCiphertext lhs = ut[static_cast<size_t>(diag_index)];
                    PhantomCiphertext rhs = enc_v.cts[static_cast<size_t>(ind)];
                    phantom_internal_rot_inplace(adapter,
                                                 rhs,
                                                 diag_index,
                                                 params.row,
                                                 params.batch_size,
                                                 params.k,
                                                 enc_v.name + "_rot_for_CCMM_V_" +
                                                     std::to_string(ind) + "_" + std::to_string(diag_index));
                    align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, "UT_" + std::to_string(diag_index), enc_v.name + "_rot_" + std::to_string(ind) +
                                           "_" + std::to_string(diag_index));
                    adapter.require_evaluator().multiply_inplace(lhs, rhs);
                    
                    if (!output_set)
                    {
                        output.cts[static_cast<size_t>(ind)] = std::move(lhs);
                        output_set = true;
                    }
                    else
                    {
                        align_scale_for_add(lhs, output.cts[static_cast<size_t>(ind)],
                                            step_name + "/UTV_" + std::to_string(ind) +
                                                "_" + std::to_string(diag_index),
                                            step_name + "/output_" + std::to_string(ind));
                        adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(ind)], lhs);
                        
                    }
                }
            }
            adapter.require_evaluator().relinearize_inplace(output.cts[static_cast<size_t>(ind)],
                                                            adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(output.cts[static_cast<size_t>(ind)]);
            
        }
    }
    else
    {
        
        cudaDeviceSynchronize();
        PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
        const int products_per_output = midqk * num_diag;
        const int internal_rot_count = midv * products_per_output;
        long long parallel_mod_switches = 0;
#pragma omp parallel num_threads(nthreads) reduction(+ : parallel_mod_switches)
        {
            PhantomCKKSEncoder phantom_encoder_local(context);
            moai::Encoder encoder_local(&context, &phantom_encoder_local);
            moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
            const int tid = omp_get_thread_num();
            auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
            for (int ind = 0; ind < midv; ++ind)
            {
                PhantomCiphertext accum;
                bool output_set = false;
                for (int i = 0; i < midqk; ++i)
                {
                    for (int j = 0; j < num_diag; ++j)
                    {
                        const int diag_index = i * num_diag + j;
                        PhantomCiphertext lhs = fhe_matrix_deep_copy_cipher_on_stream(
                            ut[static_cast<size_t>(diag_index)],
                            stream);
                        PhantomCiphertext rhs = fhe_matrix_deep_copy_cipher_on_stream(
                            enc_v.cts[static_cast<size_t>(ind)],
                            stream);
                        phantom_internal_rot_inplace_stream(adapter,
                                                            encoder_local,
                                                            evaluator_local,
                                                            stream,
                                                            rhs,
                                                            diag_index,
                                                            params.row,
                                                            params.batch_size,
                                                            params.k);
                        fhe_align_pair_for_mul_stream_quiet(context,
                                                            adapter.level_map,
                                                            lhs,
                                                            rhs,
                                                            stream,
                                                            parallel_mod_switches);
                        evaluator_local.multiply_inplace(lhs, rhs, stream);
                        if (!output_set)
                        {
                            accum = std::move(lhs);
                            output_set = true;
                        }
                        else
                        {
                            align_scale_for_add(lhs, accum,
                                                step_name + "/UTV_stream",
                                                step_name + "/output_stream");
                            evaluator_local.add_inplace(accum, lhs, stream);
                        }
                    }
                }
                evaluator_local.relinearize_inplace(accum, adapter.require_relin_keys(), stream);
                evaluator_local.rescale_to_next_inplace(accum, stream);
                output.cts[static_cast<size_t>(ind)] = std::move(accum);
            }
            cudaStreamSynchronize(stream.get_stream());
        }
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_gradient_v(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &gradient_cat_head,
                                         const FHEBlockTensor &ext_u,
                                         const GradientVOpParams &params,
                                         const std::string &out_name,
                                         const std::string &step_name = "gradient_V")
{
    
    if (gradient_cat_head.is_complex || ext_u.is_complex)
    {
        throw std::runtime_error(step_name + " requires real gradient_cat and ext_U tensors");
    }
    if (ext_u.packing != PackingType::InterleavedDiagonal)
    {
        throw std::runtime_error(step_name + " requires ext_U in interleaved diagonal packing");
    }

    std::vector<PhantomCiphertext> ut = phantom_rot_ut_grad_v(adapter, ext_u, params, step_name);
    
    return phantom_ccmm_v_batch_from_ut(adapter,
                                        ut,
                                        gradient_cat_head,
                                        params,
                                        out_name,
                                        step_name + "_CCMM_V_batch");
}

inline FHEBlockTensor phantom_gradient_k(FHEAdapterContext &adapter,
                                         const FHEBlockTensor &D,
                                         const FHEBlockTensor &K,
                                         const FHEBlockTensor &real_Q,
                                         const GradientQKOpParams &params,
                                         const std::string &out_name,
                                         const std::string &step_name = "gradient_K")
{
    
    if (D.is_complex || real_Q.is_complex)
    {
        throw std::runtime_error(step_name + " requires real D and real_Q tensors");
    }

    GradientVOpParams v_params;
    v_params.batch_size = params.batch_size;
    v_params.row = params.row;
    v_params.col = params.col;
    v_params.k = params.k;
    std::vector<PhantomCiphertext> dt = phantom_rot_ut_grad_v(adapter,
                                                              D,
                                                              v_params,
                                                              step_name);
    FHEBlockTensor DTQ = phantom_ccmm_v_batch_from_ut(adapter,
                                                      dt,
                                                      real_Q,
                                                      v_params,
                                                      out_name + "_DTQ",
                                                      step_name + "_CCMM_DTQ");
    PhantomCiphertext sum_DT = phantom_sum_diag_groups(adapter,
                                                        dt,
                                                        D.name + "_DT",
                                                        params,
                                                        false,
                                                        step_name + "_sum_DT");
    FHEBlockTensor real_K = K.is_complex
                                ? convert_complex_to_real(adapter,
                                                          K,
                                                          K.name + "_real_for_gradient_K",
                                                          2.0,
                                                          step_name + "_Complex_to_real_back_K")
                                : K;
    if (real_K.is_complex)
    {
        throw std::runtime_error(step_name + " requires K to be real or convertible to real");
    }
    if (real_K.cts.size() != DTQ.cts.size())
    {
        throw std::runtime_error(step_name + " K and DTQ ciphertext counts must match");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(real_K.cts.size());

    for (size_t i = 0; i < real_K.cts.size(); ++i)
    {
        PhantomCiphertext lhs = sum_DT;
        PhantomCiphertext rhs = real_K.cts[i];
        align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, step_name + "/sum_DT", real_K.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().multiply(lhs, rhs, output.cts[i]);
        adapter.require_evaluator().relinearize_inplace(output.cts[i], adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(output.cts[i]);

        PhantomCiphertext dtq = DTQ.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, output.cts[i], dtq, output.name + "/ct_" + std::to_string(i), DTQ.name + "/ct_" + std::to_string(i));
        align_scale_for_add(dtq, output.cts[i],
                            DTQ.name + "/ct_" + std::to_string(i),
                            output.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().sub_inplace(output.cts[i], dtq);
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline void add_internal_rotation_sum_inplace(FHEAdapterContext &adapter,
                                              PhantomCiphertext &target,
                                              int row,
                                              int batch_size,
                                              int k,
                                              const std::string &name)
{
    int rot_index = row / 2;
    while (rot_index >= 1)
    {
        PhantomCiphertext tmp = target;
        phantom_internal_rot_inplace(adapter,
                                     tmp,
                                     rot_index,
                                     row,
                                     batch_size,
                                     k,
                                     name + "_sum_internal_rot_" + std::to_string(rot_index));
        const int tmp_level = get_level(adapter.level_map, tmp);
        align_down_to(adapter.require_context(), adapter.level_map, target, tmp_level, name);
        align_scale_for_add(tmp, target,
                            name + "_rot_" + std::to_string(rot_index),
                            name);
        adapter.require_evaluator().add_inplace(target, tmp);
        
        rot_index /= 2;
    }
}

inline FHEBlockTensor phantom_ccmm_gradient_lora(FHEAdapterContext &adapter,
                                                 const FHEBlockTensor &enc_X,
                                                 const FHEBlockTensor &pre_lora_gradients,
                                                 const LoraGradientOpParams &params,
                                                 const std::string &out_name,
                                                 const std::string &step_name = "CCMM_gradient_Lora")
{
    
    if (enc_X.is_complex || pre_lora_gradients.is_complex)
    {
        throw std::runtime_error(step_name + " requires real X and pre-LoRA gradients");
    }
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0 ||
        params.rank <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " requires positive parameters");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int num_ct = params.row * params.col * params.batch_size / n;
    if (num_ct <= 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        static_cast<int>(enc_X.cts.size()) != num_ct ||
        static_cast<int>(pre_lora_gradients.cts.size()) != params.rank)
    {
        throw std::runtime_error(step_name + " ciphertext count mismatch");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.col, params.rank};
    output.packing = PackingType::RepeatedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(num_ct * params.rank));

    for (int i = 0; i < params.rank; ++i)
    {
        for (int j = 0; j < num_ct; ++j)
        {
            PhantomCiphertext lhs = enc_X.cts[static_cast<size_t>(j)];
            PhantomCiphertext rhs = pre_lora_gradients.cts[static_cast<size_t>(i)];
            align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, enc_X.name + "/ct_" + std::to_string(j), pre_lora_gradients.name + "/rank_" + std::to_string(i));
            PhantomCiphertext product;
            adapter.require_evaluator().multiply(lhs, rhs, product);
            adapter.require_evaluator().relinearize_inplace(product, adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(product);

            add_internal_rotation_sum_inplace(adapter,
                                              product,
                                              params.row,
                                              params.batch_size,
                                              params.k,
                                              step_name + "_out_" +
                                                  std::to_string(i) + "_" + std::to_string(j));
            output.cts[static_cast<size_t>(i * num_ct + j)] = std::move(product);
        }
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_gradient_lora_a(FHEAdapterContext &adapter,
                                              const FHEBlockTensor &enc_X,
                                              const FHEBlockTensor &gradient,
                                              const FHEBlockTensor &B_complex,
                                              const LoraGradientOpParams &params,
                                              const std::string &out_name,
                                              const std::string &step_name = "gradient_A")
{
    
    if (!B_complex.is_complex)
    {
        throw std::runtime_error(step_name + " requires B in complex packing");
    }

    FHEBlockTensor real_B = convert_complex_to_real(adapter,
                                                    B_complex,
                                                    B_complex.name + "_real_for_gradient_A",
                                                    1.0,
                                                    step_name + "_Complex_to_real_B");

    MatrixOpParams qbt_params;
    qbt_params.batch_size = params.batch_size;
    qbt_params.row_X = params.row;
    qbt_params.col_X = params.col;
    qbt_params.row_W = params.col;
    qbt_params.col_W = params.rank;
    qbt_params.k = params.k;
    FHEBlockTensor qbt = phantom_ccmm_lora_a(adapter,
                                            gradient,
                                            real_B,
                                            qbt_params,
                                            step_name + "_QBT",
                                            step_name + "_CCMM_LoRA_A_QBT");

    return phantom_ccmm_gradient_lora(adapter,
                                      enc_X,
                                      qbt,
                                      params,
                                      out_name,
                                      step_name + "_CCMM_gradient_Lora");
}

inline FHEBlockTensor phantom_gradient_lora_b(FHEAdapterContext &adapter,
                                              const FHEBlockTensor &XA,
                                              const FHEBlockTensor &gradient,
                                              const LoraGradientOpParams &params,
                                              const std::string &out_name,
                                              const std::string &step_name = "gradient_B",
                                              PackingType output_packing = PackingType::InterleavedColumnComplex)
{
    
    if (XA.is_complex || gradient.is_complex)
    {
        throw std::runtime_error(step_name + " requires real XA and real gradient tensors");
    }
    if (params.rank != 2 || static_cast<int>(XA.cts.size()) != params.rank)
    {
        throw std::runtime_error(step_name + " currently matches CPU rank-2 XA complex repack");
    }

    PhantomCiphertext complex_XA = XA.cts[0];
    PhantomCiphertext imag_XA = XA.cts[1];
    PhantomPlaintext i_plain = encode_complex_constant_like(adapter,
                                                            imag_XA,
                                                            std::complex<double>(0.0, 1.0));
    adapter.require_evaluator().multiply_plain_inplace(imag_XA, i_plain);
    adapter.require_evaluator().rescale_to_next_inplace(imag_XA);
    
    const int imag_level = get_level(adapter.level_map, imag_XA);
    align_down_to(adapter.require_context(), adapter.level_map, complex_XA, imag_level, step_name + "/complex_XA_real");
    align_scale_for_add(imag_XA, complex_XA,
                        step_name + "/complex_XA_imag",
                        step_name + "/complex_XA_real");
    adapter.require_evaluator().add_inplace(complex_XA, imag_XA);

    FHEBlockTensor complex_output;
    complex_output.name = out_name + "_complex_before_repack";
    complex_output.logical_shape = {params.rank, params.col};
    complex_output.packing = PackingType::RepeatedRow;
    complex_output.is_complex = true;
    complex_output.cts.resize(gradient.cts.size());

    for (size_t i = 0; i < gradient.cts.size(); ++i)
    {
        PhantomCiphertext lhs = complex_XA;
        PhantomCiphertext rhs = gradient.cts[i];
        align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, step_name + "/complex_XA", gradient.name + "/ct_" + std::to_string(i));
        PhantomCiphertext product;
        adapter.require_evaluator().multiply(lhs, rhs, product);
        adapter.require_evaluator().relinearize_inplace(product, adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(product);
        
        add_internal_rotation_sum_inplace(adapter,
                                          product,
                                          params.row,
                                          params.batch_size,
                                          params.k,
                                          step_name + "_complex_out_" + std::to_string(i));
        complex_output.cts[i] = std::move(product);
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, complex_output);

    FHEBlockTensor real_parts = convert_complex_to_real(adapter,
                                                        complex_output,
                                                        out_name + "_real_parts",
                                                        1.0,
                                                        step_name + "_Complex_to_real_output");

    FHEBlockTensor rank_major = make_like_metadata(real_parts, out_name + "_rank_major_lanes");
    for (int rank = 0; rank < params.rank; ++rank)
        for (size_t tile = 0; tile < gradient.cts.size(); ++tile)
            rank_major.cts.push_back(real_parts.cts.at(2*tile + rank));
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, rank_major);
    auto layout = make_like_metadata(complex_output, out_name);
    layout.packing = output_packing;
    layout.cts.resize(gradient.cts.size());
    FHEBlockTensor output = pack_independent_complex_lanes(adapter, rank_major, layout, out_name);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_lora_b_real(FHEAdapterContext &adapter,
                                               const FHEBlockTensor &enc_X,
                                               const FHEBlockTensor &W,
                                               const MatrixOpParams &params,
                                               const std::string &out_name,
                                               const std::string &step_name = "CCMM-B-real")
{
    
    validate_matrix_params(params);
    if (enc_X.is_complex || W.is_complex)
    {
        throw std::runtime_error(step_name + " requires real enc_X and W tensors");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = static_cast<int>(enc_X.cts.size());
    const int ed = params.row_X * params.col_W * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row_X * params.col_W * params.batch_size % n != 0 ||
        static_cast<int>(W.cts.size()) != mid * ed)
    {
        throw std::runtime_error(step_name + " ciphertext count mismatch");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(ed));

    for (int i = 0; i < ed; ++i)
    {
        bool set = false;
        for (int j = 0; j < mid; ++j)
        {
            PhantomCiphertext lhs = enc_X.cts[static_cast<size_t>(j)];
            PhantomCiphertext rhs = W.cts[static_cast<size_t>(j * ed + i)];
            align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, enc_X.name + "/ct_" + std::to_string(j), W.name + "/ct_" + std::to_string(j * ed + i));
            PhantomCiphertext product;
            adapter.require_evaluator().multiply(lhs, rhs, product);
            
            if (!set)
            {
                output.cts[static_cast<size_t>(i)] = std::move(product);
                set = true;
            }
            else
            {
                align_scale_for_add(product, output.cts[static_cast<size_t>(i)],
                                    step_name + "/product_" + std::to_string(j),
                                    step_name + "/output_" + std::to_string(i));
                adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(i)], product);
                
            }
        }
        adapter.require_evaluator().relinearize_inplace(output.cts[static_cast<size_t>(i)],
                                                        adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(output.cts[static_cast<size_t>(i)]);
        
    }

    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_gradient_x_part1(FHEAdapterContext &adapter,
                                               const FHEBlockTensor &gradient,
                                               const FHEBlockTensor &A,
                                               const FHEBlockTensor &B_complex,
                                               const std::vector<std::vector<double>> &W,
                                               const LoraGradientOpParams &params,
                                               const std::string &out_name,
                                               const std::string &step_name = "gradient_x_part1")
{
    
    if (gradient.is_complex || A.is_complex || !B_complex.is_complex)
    {
        throw std::runtime_error(step_name + " requires real gradient/A and complex B");
    }
    if (W.empty())
    {
        throw std::runtime_error(step_name + " requires plaintext W");
    }

    MatrixOpParams qwt_params;
    qwt_params.batch_size = params.batch_size;
    qwt_params.row_X = params.row;
    qwt_params.col_X = params.col;
    qwt_params.row_W = params.col;
    qwt_params.col_W = params.col;
    qwt_params.k = params.k;
    FHEBlockTensor qwt = phantom_pcmm_real(adapter,
                                           gradient,
                                           transpose_matrix(W),
                                           qwt_params,
                                           step_name + "_QWT",
                                           step_name + "_PCMM_WT");

    FHEBlockTensor real_B = convert_complex_to_real(adapter,
                                                    B_complex,
                                                    B_complex.name + "_real_for_gradient_x",
                                                    1.0,
                                                    step_name + "_Complex_to_real_B");

    MatrixOpParams qbt_params;
    qbt_params.batch_size = params.batch_size;
    qbt_params.row_X = params.row;
    qbt_params.col_X = params.col;
    qbt_params.row_W = params.col;
    qbt_params.col_W = params.rank;
    qbt_params.k = params.k;
    FHEBlockTensor qbt = phantom_ccmm_lora_a(adapter,
                                            gradient,
                                            real_B,
                                            qbt_params,
                                            step_name + "_QBT",
                                            step_name + "_CCMM_LoRA_A_QBT");

    MatrixOpParams qbtat_params;
    qbtat_params.batch_size = params.batch_size;
    qbtat_params.row_X = params.row;
    qbtat_params.col_X = params.rank;
    qbtat_params.row_W = params.rank;
    qbtat_params.col_W = params.col;
    qbtat_params.k = params.k;
    FHEBlockTensor qbtat = phantom_ccmm_lora_b_real(adapter,
                                                    qbt,
                                                    A,
                                                    qbtat_params,
                                                    step_name + "_QBTAT",
                                                    step_name + "_CCMM_LoRA_B_QBTAT");

    require_same_ct_count(qwt, qbtat, step_name);
    FHEBlockTensor output = qwt;
    output.name = out_name;
    
    for (size_t i = 0; i < output.cts.size(); ++i)
    {
        PhantomCiphertext addend = qbtat.cts[i];
        align_pair_for_add(adapter.require_context(), adapter.level_map, output.cts[i], addend, qwt.name + "/ct_" + std::to_string(i), qbtat.name + "/ct_" + std::to_string(i));
        align_scale_for_add(addend, output.cts[i],
                            qbtat.name + "/ct_" + std::to_string(i),
                            qwt.name + "/ct_" + std::to_string(i));
        adapter.require_evaluator().add_inplace(output.cts[i], addend);
        
    }

    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_gradient_x_sum(FHEAdapterContext &adapter,
                                             const FHEBlockTensor &gradient_ln1,
                                             const std::vector<FHEBlockTensor> &parts,
                                             const std::string &out_name,
                                             const std::string &step_name = "gradient_X_sum")
{
    
    if (parts.empty())
    {
        throw std::runtime_error(step_name + " requires at least one branch part");
    }
    FHEBlockTensor output = gradient_ln1;
    output.name = out_name;
    
    for (const auto &part : parts)
    {
        
        require_same_ct_count(output, part, step_name);
        for (size_t i = 0; i < output.cts.size(); ++i)
        {
            PhantomCiphertext addend = part.cts[i];
            align_pair_for_add(adapter.require_context(), adapter.level_map, output.cts[i], addend, output.name + "/ct_" + std::to_string(i), part.name + "/ct_" + std::to_string(i));
            align_scale_for_add(addend, output.cts[i],
                                part.name + "/ct_" + std::to_string(i),
                                output.name + "/ct_" + std::to_string(i));
            adapter.require_evaluator().add_inplace(output.cts[i], addend);
            
        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    }

    output.logical_shape = gradient_ln1.logical_shape;
    output.packing = gradient_ln1.packing;
    output.is_complex = false;
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

}
