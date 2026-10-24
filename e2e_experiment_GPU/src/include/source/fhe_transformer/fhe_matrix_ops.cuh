#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "source/fhe_transformer/fhe_plaintext_packing.cuh"

namespace fhe_transformer {

inline int fhe_stream_parallel_threads(int work_items,int default_cap=4) {
    return std::max(1,std::min({omp_get_max_threads(),default_cap,work_items}));
}

inline bool fhe_force_pcmm_cached_rotations(){return true;}

inline std::vector<std::unique_ptr<phantom::util::cuda_stream_wrapper>> &fhe_stream_parallel_pool()
{
    static std::vector<std::unique_ptr<phantom::util::cuda_stream_wrapper>> pool;
    return pool;
}

inline phantom::util::cuda_stream_wrapper &fhe_stream_parallel_stream(int tid, int nthreads)
{
    if (nthreads <= 1)
    {
        return *phantom::util::global_variables::default_stream;
    }
    auto &pool = fhe_stream_parallel_pool();
    while (pool.size() < static_cast<size_t>(nthreads))
    {
        pool.emplace_back(std::make_unique<phantom::util::cuda_stream_wrapper>());
    }
    return *pool[static_cast<size_t>(tid)];
}

inline PhantomCiphertext fhe_matrix_deep_copy_cipher_on_stream(
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

inline void publish_matrix_output(FHEBlockTensor &output)
{
    const auto &stream=*phantom::util::global_variables::default_stream;
    for(auto &ct:output.cts) {
        if(ct.data_ptr().get_stream()==stream.get_stream())continue;
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(ct.data_ptr().get_stream()));
        auto source=std::move(ct);
        ct=fhe_matrix_deep_copy_cipher_on_stream(source,stream);
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream.get_stream()));
    }
}

inline void precompute_pcmm_baby_step_rotations(FHEAdapterContext &adapter,
                                                const FHEBlockTensor &enc_X,
                                                int mid,
                                                int N1,
                                                int rotate_stride,
                                                std::vector<PhantomCiphertext> &ct_N1,
                                                const std::string &step_name)
{
    const int work_items = mid * N1;
    const int nthreads = 1;
    if (nthreads <= 1)
    {
        for (int j = 0; j < mid; ++j)
        {
            for (int q = 0; q < N1; ++q)
            {
                adapter.require_evaluator().rotate_vector(const_cast<PhantomCiphertext &>(enc_X.cts[j]),
                                                          q * rotate_stride,
                                                          adapter.require_galois_keys(),
                                                          ct_N1[static_cast<size_t>(j * N1 + q)]);
                
            }
        }
        return;
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
        for (int work = 0; work < work_items; ++work)
        {
            const int j = work / N1;
            const int q = work % N1;
            PhantomCiphertext src = fhe_matrix_deep_copy_cipher_on_stream(
                enc_X.cts[static_cast<size_t>(j)],
                stream);
            evaluator_local.rotate_vector(src,
                                          q * rotate_stride,
                                          adapter.require_galois_keys(),
                                          ct_N1[static_cast<size_t>(work)],
                                          stream);
        }
        cudaStreamSynchronize(stream.get_stream());
    }
    
}

inline int fhe_align_down_to_stream_quiet(const PhantomContext &context,
                                          const FHELevelMap &map,
                                          PhantomCiphertext &ct,
                                          int target_level,
                                          const phantom::util::cuda_stream_wrapper &stream)
{
    const int current_level = get_level(map, ct);
    if (current_level == target_level)
    {
        return 0;
    }
    if (current_level < target_level)
    {
        throw std::runtime_error("stream-parallel align cannot promote ciphertext from level " +
                                 std::to_string(current_level) + " to " + std::to_string(target_level));
    }
    const size_t target_chain_index = map.chain_index_for_logical_level(context, target_level);
    ::mod_switch_to_inplace(context, ct, target_chain_index, stream);
    return current_level - target_level;
}

inline int fhe_align_pair_for_mul_stream_quiet(const PhantomContext &context,
                                               const FHELevelMap &map,
                                               PhantomCiphertext &a,
                                               PhantomCiphertext &b,
                                               const phantom::util::cuda_stream_wrapper &stream,
                                               long long &mod_switch_count)
{
    const int target = std::min(get_level(map, a), get_level(map, b));
    mod_switch_count += fhe_align_down_to_stream_quiet(context, map, a, target, stream);
    mod_switch_count += fhe_align_down_to_stream_quiet(context, map, b, target, stream);
    assert_same_level_for_mul(map, a, b, "stream_lhs", "stream_rhs");
    return target;
}

struct MatrixOpParams {
    int batch_size = 1;
    int col_X = 0;
    int row_X = 0;
    int col_W = 0;
    int row_W = 0;
    int k = 1;
};

inline void add_column_dense_bias(FHEAdapterContext &adapter, FHEBlockTensor &value,
                                  const std::vector<double> &bias, const MatrixOpParams &params)
{
    if (bias.empty()) return;
    if (static_cast<int>(bias.size()) != params.col_W)
        throw std::invalid_argument("Dense bias dimension mismatch");
    const int slots = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    std::vector<std::vector<double>> matrix(params.row_X, bias);
    if (value.is_complex) {
        std::vector<std::vector<std::vector<double>>> batched(params.batch_size, matrix);
        auto packed = modified_col_packing_batch_complex_pt(
            batched, params.batch_size, params.row_X, params.col_W, params.k, slots);
        if (packed.size() != value.cts.size()) throw std::runtime_error("Complex dense bias packing mismatch");
        for (size_t i = 0; i < packed.size(); ++i) {
            PhantomPlaintext p;
            encode_values(adapter, packed[i], value.cts[i].params_id(), value.cts[i].scale(), p);
            adapter.require_evaluator().add_plain_inplace(value.cts[i], p);
        }
    } else {
        auto packed = modified_col_packing_batch_pt(matrix, params.batch_size,
                                                   params.row_X, params.col_W, params.k, slots);
        if (packed.size() != value.cts.size()) throw std::runtime_error("Real dense bias packing mismatch");
        for (size_t i = 0; i < packed.size(); ++i) {
            PhantomPlaintext p;
            encode_values(adapter, packed[i], value.cts[i].params_id(), value.cts[i].scale(), p);
            adapter.require_evaluator().add_plain_inplace(value.cts[i], p);
        }
    }
    
}

struct AttentionOpParams {
    int batch_size = 1;
    int row = 0;
    int col = 0;
    int k = 1;
    double distance_scale = 1.0;
};

struct AttentionValueOpParams {
    int batch_size = 1;
    int qk_row = 0;
    int v_col = 0;
    int v_row = 0;
    int k = 1;
};

inline std::pair<int, int> choose_bsgs_steps(int total)
{
    if (total <= 0)
    {
        throw std::runtime_error("choose_bsgs_steps requires positive total");
    }
    int n1 = static_cast<int>(std::sqrt(static_cast<double>(total)));
    while (n1 > 1 && total % n1 != 0)
    {
        --n1;
    }
    return {n1, total / n1};
}

inline std::vector<std::vector<double>> transpose_matrix(const std::vector<std::vector<double>> &input)
{
    if (input.empty() || input.front().empty())
    {
        throw std::runtime_error("transpose_matrix requires a non-empty matrix");
    }
    const size_t rows = input.size();
    const size_t cols = input.front().size();
    for (const auto &row : input)
    {
        if (row.size() != cols)
        {
            throw std::runtime_error("transpose_matrix requires a rectangular matrix");
        }
    }

    std::vector<std::vector<double>> output(cols, std::vector<double>(rows, 0.0));
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            output[j][i] = input[i][j];
        }
    }
    return output;
}

inline void validate_matrix_params(const MatrixOpParams &params)
{
    if (params.col_X != params.row_W)
    {
        throw std::runtime_error("bad dimensions: col_X must equal row_W");
    }
    if (params.batch_size <= 0 || params.row_X <= 0 || params.col_X <= 0 ||
        params.col_W <= 0 || params.row_W <= 0 || params.k <= 0)
    {
        throw std::runtime_error("matrix op parameters must be positive");
    }
}

inline FHEBlockTensor phantom_pcmm_real(FHEAdapterContext &adapter,
                                        const FHEBlockTensor &enc_X,
                                        const std::vector<std::vector<double>> &W,
                                        const MatrixOpParams &params,
                                        const std::string &out_name,
                                        const std::string &step_name = "PCMM-real")
{
    
    validate_matrix_params(params);
    if (enc_X.is_complex)
    {
        throw std::runtime_error(step_name + " requires real interleaved-column input");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int n2 = n / params.batch_size;
    const int mid = params.row_X * params.col_X * params.batch_size / n;
    const int ed = params.row_X * params.col_W * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row_X * params.col_X * params.batch_size % n != 0 ||
        params.row_X * params.col_W * params.batch_size % n != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not produce integral ciphertext counts");
    }
    if (static_cast<int>(enc_X.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " input ciphertext count mismatch");
    }
    if (static_cast<int>(W.size()) != params.row_W || static_cast<int>(W.front().size()) != params.col_W)
    {
        throw std::runtime_error(step_name + " plaintext W shape mismatch");
    }

    const auto [N1, N2] = choose_bsgs_steps(n2 / params.row_X);

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(ed));

    const bool force_cached_rotations = fhe_force_pcmm_cached_rotations();
    const bool stream_rotations = (mid * N1 > ed * N2) && !force_cached_rotations;
    if (force_cached_rotations && mid * N1 > ed * N2)
    {
        
    }
    if (stream_rotations)
    {

        std::vector<PhantomCiphertext> partial(static_cast<size_t>(ed * N2));
        std::vector<char> partial_set(static_cast<size_t>(ed * N2), 0);

        for (int j = 0; j < mid; ++j)
        {
            for (int q = 0; q < N1; ++q)
            {
                PhantomCiphertext rotated;
                adapter.require_evaluator().rotate_vector(const_cast<PhantomCiphertext &>(enc_X.cts[j]),
                                                          q * params.row_X * params.batch_size,
                                                          adapter.require_galois_keys(),
                                                          rotated);

                for (int l = 0; l < ed; ++l)
                {
                    for (int p = 0; p < N2; ++p)
                    {
                        std::vector<double> d(static_cast<size_t>(n), 0.0);
                        const int i = p * N1 + q;
                        for (int ind1 = 0; ind1 < (n2 / params.row_X); ++ind1)
                        {
                            for (int ind2 = 0; ind2 < params.row_X; ++ind2)
                            {
                                for (int ind3 = 0; ind3 < params.batch_size; ++ind3)
                                {
                                    const int slot = ind1 * params.row_X * params.batch_size +
                                                     ind2 * params.batch_size + ind3;
                                    const int w_row = j * n2 / params.row_X +
                                                      (i + ind1) % (n2 / params.row_X);
                                    const int w_col = l * n2 / params.row_X +
                                                      ind1 % (n2 / params.row_X);
                                    d[static_cast<size_t>(slot)] =
                                        W[static_cast<size_t>(w_row)]
                                         [static_cast<size_t>(w_col)];
                                }
                            }
                        }

                        PhantomPlaintext pt_d;
                        encode_values(adapter,
                            vec_left_rot(d, -p * N1 * params.row_X * params.batch_size),
                            rotated.params_id(),
                            rotated.scale(),
                            pt_d);
                        PhantomCiphertext ct_d;
                        adapter.require_evaluator().multiply_plain(rotated, pt_d, ct_d);

                        const size_t partial_index = static_cast<size_t>(l * N2 + p);
                        if (!partial_set[partial_index])
                        {
                            partial[partial_index] = std::move(ct_d);
                            partial_set[partial_index] = 1;
                        }
                        else
                        {
                            adapter.require_evaluator().add_inplace(partial[partial_index], ct_d);
                            
                        }
                    }
                }
            }
        }

        for (int l = 0; l < ed; ++l)
        {
            bool output_set = false;
            for (int p = 0; p < N2; ++p)
            {
                const size_t partial_index = static_cast<size_t>(l * N2 + p);
                if (!partial_set[partial_index])
                {
                    throw std::runtime_error(step_name + " streaming PCMM missing partial accumulator");
                }
                PhantomCiphertext tmp = std::move(partial[partial_index]);
                adapter.require_evaluator().rotate_vector_inplace(tmp,
                                                                  p * N1 * params.row_X * params.batch_size,
                                                                  adapter.require_galois_keys());
                
                if (!output_set)
                {
                    output.cts[static_cast<size_t>(l)] = std::move(tmp);
                    output_set = true;
                }
                else
                {
                    adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(l)], tmp);
                    
                }
            }
            adapter.require_evaluator().rescale_to_next_inplace(output.cts[static_cast<size_t>(l)]);
            
        }
    }
    else
    {
        std::vector<PhantomCiphertext> ct_N1(static_cast<size_t>(mid * N1));
        precompute_pcmm_baby_step_rotations(adapter,
                                            enc_X,
                                            mid,
                                            N1,
                                            params.row_X * params.batch_size,
                                            ct_N1,
                                            step_name);
        cudaDeviceSynchronize();

        const int nthreads = fhe_stream_parallel_threads(ed);
        
        PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
        {
            PhantomCKKSEncoder phantom_encoder_local(context);
            moai::Encoder encoder_local(&context, &phantom_encoder_local);
            moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
            const int tid = omp_get_thread_num();
            auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
            for (int l = 0; l < ed; ++l)
            {
                bool output_set = false;
                for (int p = 0; p < N2; ++p)
                {
                    PhantomCiphertext tmp;
                    bool tmp_set = false;
                    for (int j = 0; j < mid; ++j)
                    {
                        for (int q = 0; q < N1; ++q)
                        {
                            std::vector<double> d(static_cast<size_t>(n), 0.0);
                            const int i = p * N1 + q;
                            for (int ind1 = 0; ind1 < (n2 / params.row_X); ++ind1)
                            {
                                for (int ind2 = 0; ind2 < params.row_X; ++ind2)
                                {
                                    for (int ind3 = 0; ind3 < params.batch_size; ++ind3)
                                    {
                                        const int slot = ind1 * params.row_X * params.batch_size +
                                                         ind2 * params.batch_size + ind3;
                                        const int w_row = j * n2 / params.row_X +
                                                          (i + ind1) % (n2 / params.row_X);
                                        const int w_col = l * n2 / params.row_X +
                                                          ind1 % (n2 / params.row_X);
                                        d[static_cast<size_t>(slot)] = W[static_cast<size_t>(w_row)]
                                                                        [static_cast<size_t>(w_col)];
                                    }
                                }
                            }

                            PhantomPlaintext pt_d;
                            encode_values(encoder_local, vec_left_rot(d, -p * N1 * params.row_X * params.batch_size), ct_N1[static_cast<size_t>(j * N1 + q)].params_id(), ct_N1[static_cast<size_t>(j * N1 + q)].scale(), pt_d, stream);
                            PhantomCiphertext ct_d;
                            PhantomCiphertext ct_src = fhe_matrix_deep_copy_cipher_on_stream(
                                ct_N1[static_cast<size_t>(j * N1 + q)],
                                stream);
                            evaluator_local.multiply_plain(ct_src,
                                                           pt_d,
                                                           ct_d,
                                                           stream);
                            if (!tmp_set)
                            {
                                tmp = std::move(ct_d);
                                tmp_set = true;
                            }
                            else
                            {
                                evaluator_local.add_inplace(tmp, ct_d, stream);
                            }
                        }
                    }
                    evaluator_local.rotate_vector_inplace(tmp,
                                                          p * N1 * params.row_X * params.batch_size,
                                                          adapter.require_galois_keys(),
                                                          stream);
                    if (!output_set)
                    {
                        output.cts[static_cast<size_t>(l)] = std::move(tmp);
                        output_set = true;
                    }
                    else
                    {
                        evaluator_local.add_inplace(output.cts[static_cast<size_t>(l)], tmp, stream);
                    }
                }
                evaluator_local.rescale_to_next_inplace(output.cts[static_cast<size_t>(l)], stream);
            }
            cudaStreamSynchronize(stream.get_stream());
        }
        
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline void phantom_internal_rot_inplace(FHEAdapterContext &adapter,
                                         PhantomCiphertext &x,
                                         int rot_index,
                                         int vec_size,
                                         int batch_size,
                                         int k,
                                         const std::string &name)
{
    (void)k;
    if (vec_size <= 0 || batch_size <= 0)
    {
        throw std::runtime_error("internal rotation requires positive vec_size and batch_size");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    if (n % (vec_size * batch_size) != 0)
    {
        throw std::runtime_error("internal rotation dimensions do not divide slot count");
    }
    if (rot_index < 0 || rot_index > vec_size)
    {
        throw std::runtime_error("internal rotation rot_index out of range for " + name);
    }

    PhantomCiphertext copy = x;
    const int vec_num = n / (vec_size * batch_size);
    std::vector<double> mask(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < vec_num; ++i)
    {
        for (int j = 0; j < rot_index * batch_size; ++j)
        {
            mask[static_cast<size_t>(i * vec_size * batch_size + j)] = 1.0;
        }
    }

    PhantomPlaintext encoded_mask;
    encode_values(adapter, mask, x.params_id(), static_cast<double>(adapter.require_context().get_context_data(x.params_id()).parms().coeff_modulus().back().value()), encoded_mask);
    adapter.require_evaluator().multiply_plain_inplace(x, encoded_mask);
    adapter.require_evaluator().rescale_to_next_inplace(x);
    adapter.require_evaluator().mod_switch_to_inplace(copy, x.params_id());
    x.scale() = copy.scale();
    adapter.require_evaluator().sub_inplace(copy, x);
    adapter.require_evaluator().rotate_vector_inplace(copy,
                                                      rot_index * batch_size,
                                                      adapter.require_galois_keys());
    adapter.require_evaluator().rotate_vector_inplace(x,
                                                      -1 * (vec_size - rot_index) * batch_size,
                                                      adapter.require_galois_keys());
    adapter.require_evaluator().add_inplace(x, copy);

}

inline void phantom_internal_rot_inplace_stream(FHEAdapterContext &adapter,
                                                moai::Encoder &encoder_local,
                                                moai::Evaluator &evaluator_local,
                                                const phantom::util::cuda_stream_wrapper &stream,
                                                PhantomCiphertext &x,
                                                int rot_index,
                                                int vec_size,
                                                int batch_size,
                                                int k)
{
    (void)k;
    if (vec_size <= 0 || batch_size <= 0)
    {
        throw std::runtime_error("internal rotation requires positive vec_size and batch_size");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    if (n % (vec_size * batch_size) != 0)
    {
        throw std::runtime_error("internal rotation dimensions do not divide slot count");
    }
    if (rot_index < 0 || rot_index > vec_size)
    {
        throw std::runtime_error("internal rotation rot_index out of range");
    }

    PhantomCiphertext copy = fhe_matrix_deep_copy_cipher_on_stream(x, stream);
    const int vec_num = n / (vec_size * batch_size);
    std::vector<double> mask(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < vec_num; ++i)
    {
        for (int j = 0; j < rot_index * batch_size; ++j)
        {
            mask[static_cast<size_t>(i * vec_size * batch_size + j)] = 1.0;
        }
    }

    PhantomPlaintext encoded_mask;
    encode_values(encoder_local, mask, x.params_id(), static_cast<double>(adapter.require_context().get_context_data(x.params_id()).parms().coeff_modulus().back().value()), encoded_mask, stream);
    evaluator_local.multiply_plain_inplace(x, encoded_mask, stream);
    evaluator_local.rescale_to_next_inplace(x, stream);
    evaluator_local.mod_switch_to_inplace(copy, x.params_id(), stream);
    x.scale() = copy.scale();
    evaluator_local.sub_inplace(copy, x, stream);
    evaluator_local.rotate_vector_inplace(copy,
                                          rot_index * batch_size,
                                          adapter.require_galois_keys(),
                                          stream);
    evaluator_local.rotate_vector_inplace(x,
                                          -1 * (vec_size - rot_index) * batch_size,
                                          adapter.require_galois_keys(),
                                          stream);
    evaluator_local.add_inplace(x, copy, stream);
}

inline FHEBlockTensor phantom_pcmm_complex(FHEAdapterContext &adapter,
                                           const FHEBlockTensor &enc_X,
                                           const std::vector<std::vector<double>> &W,
                                           const MatrixOpParams &params,
                                           const std::string &out_name,
                                           bool selfoutput_variant = false,
                                           const std::string &step_name = "PCMM-complex")
{
    
    validate_matrix_params(params);

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int n2 = n / params.batch_size;
    const int mid = params.row_X * params.col_X * params.batch_size / n;
    const int ed = params.row_X * params.col_W * params.batch_size / (2 * n);
    if (mid <= 0 || ed <= 0 ||
        params.row_X * params.col_X * params.batch_size % n != 0 ||
        params.row_X * params.col_W * params.batch_size % (2 * n) != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not produce integral ciphertext counts");
    }
    if (static_cast<int>(enc_X.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " input ciphertext count mismatch");
    }
    if (static_cast<int>(W.size()) != params.row_W || static_cast<int>(W[0].size()) != params.col_W)
    {
        throw std::runtime_error(step_name + " plaintext W shape mismatch");
    }

    const auto [N1, N2] = choose_bsgs_steps(n2 / params.row_X);

    std::vector<PhantomCiphertext> ct_N1(static_cast<size_t>(mid * N1));
    precompute_pcmm_baby_step_rotations(adapter,
                                        enc_X,
                                        mid,
                                        N1,
                                        params.row_X * params.batch_size,
                                        ct_N1,
                                        step_name);
    cudaDeviceSynchronize();

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::InterleavedColumnComplex;
    output.is_complex = true;
    output.cts.resize(ed);

    const int nthreads = fhe_stream_parallel_threads(ed);
    
    PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
    {
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Encoder encoder_local(&context, &phantom_encoder_local);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
        const int tid = omp_get_thread_num();
        auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
        for (int l = 0; l < ed; ++l)
        {
            bool output_set = false;
            for (int p = 0; p < N2; ++p)
            {
                PhantomCiphertext tmp;
                bool tmp_set = false;
                for (int j = 0; j < mid; ++j)
                {
                    for (int q = 0; q < N1; ++q)
                    {
                        std::vector<std::complex<double>> d(n, 0.0);
                        const int i = p * N1 + q;
                        for (int ind1 = 0; ind1 < (n2 / params.row_X); ++ind1)
                        {
                            for (int ind2 = 0; ind2 < params.row_X; ++ind2)
                            {
                                for (int ind3 = 0; ind3 < params.batch_size; ++ind3)
                                {
                                    const int slot = ind1 * params.row_X * params.batch_size +
                                                     ind2 * params.batch_size + ind3;
                                    const int w_row = j * n2 / params.row_X +
                                                      (i + ind1) % (n2 / params.row_X);
                                    const int w_col_real = 2 * l * n2 / params.row_X +
                                                           ind1 % (n2 / params.row_X);
                                    const int w_col_imag = (2 * l + 1) * n2 / params.row_X +
                                                           ind1 % (n2 / params.row_X);
                                    if (!selfoutput_variant || (j % 2) == 0)
                                    {
                                        const double scale = selfoutput_variant ? 0.5 : 1.0;
                                        d[slot].real(scale * W[w_row][w_col_real]);
                                        d[slot].imag(scale * W[w_row][w_col_imag]);
                                    }
                                    else
                                    {
                                        d[slot].imag(-0.5 * W[w_row][w_col_real]);
                                        d[slot].real(0.5 * W[w_row][w_col_imag]);
                                    }
                                }
                            }
                        }

                        PhantomPlaintext pt_d;
                        encode_values(encoder_local, vec_left_rot_complex(d, -p * N1 * params.row_X * params.batch_size), ct_N1[j * N1 + q].params_id(), ct_N1[j * N1 + q].scale(), pt_d, stream);
                        PhantomCiphertext ct_d;
                        PhantomCiphertext ct_src = fhe_matrix_deep_copy_cipher_on_stream(ct_N1[j * N1 + q], stream);
                        evaluator_local.multiply_plain(ct_src, pt_d, ct_d, stream);
                        if (!tmp_set)
                        {
                            tmp = std::move(ct_d);
                            tmp_set = true;
                        }
                        else
                        {
                            evaluator_local.add_inplace(tmp, ct_d, stream);
                        }
                    }
                }
                evaluator_local.rotate_vector_inplace(tmp,
                                                      p * N1 * params.row_X * params.batch_size,
                                                      adapter.require_galois_keys(),
                                                      stream);
                if (!output_set)
                {
                    output.cts[l] = std::move(tmp);
                    output_set = true;
                }
                else
                {
                    evaluator_local.add_inplace(output.cts[l], tmp, stream);
                }
            }
            evaluator_local.rescale_to_next_inplace(output.cts[l], stream);
        }
        cudaStreamSynchronize(stream.get_stream());
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_pcmm_selfoutput_real(FHEAdapterContext &adapter,
                                                   const FHEBlockTensor &enc_X,
                                                   const std::vector<std::vector<double>> &W,
                                                   const MatrixOpParams &params,
                                                   const std::string &out_name,
                                                   const std::string &step_name = "PCMM-selfoutput-complex")
{
    
    validate_matrix_params(params);
    if (!enc_X.is_complex || enc_X.packing != PackingType::InterleavedColumnMix)
    {
        throw std::runtime_error(step_name + " requires InterleavedColumnMix complex input from merged attention heads");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int n2 = n / params.batch_size;
    const int mid = params.row_X * params.col_X * params.batch_size / n;
    const int ed = params.row_X * params.col_W * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row_X * params.col_X * params.batch_size % n != 0 ||
        params.row_X * params.col_W * params.batch_size % n != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not produce integral ciphertext counts");
    }
    if (static_cast<int>(enc_X.cts.size()) != mid)
    {
        throw std::runtime_error(step_name + " input ciphertext count mismatch");
    }
    if (static_cast<int>(W.size()) != params.row_W || static_cast<int>(W[0].size()) != params.col_W)
    {
        throw std::runtime_error(step_name + " plaintext W shape mismatch");
    }

    const auto [N1, N2] = choose_bsgs_steps(n2 / params.row_X);

    std::vector<PhantomCiphertext> ct_N1(static_cast<size_t>(mid * N1));
    precompute_pcmm_baby_step_rotations(adapter,
                                        enc_X,
                                        mid,
                                        N1,
                                        params.row_X * params.batch_size,
                                        ct_N1,
                                        step_name);
    cudaDeviceSynchronize();

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(ed);

    const int nthreads = fhe_stream_parallel_threads(ed);
    
    PhantomContext &context = const_cast<PhantomContext &>(adapter.require_context());
#pragma omp parallel num_threads(nthreads)
    {
        PhantomCKKSEncoder phantom_encoder_local(context);
        moai::Encoder encoder_local(&context, &phantom_encoder_local);
        moai::Evaluator evaluator_local(&context, &phantom_encoder_local);
        const int tid = omp_get_thread_num();
        auto &stream = fhe_stream_parallel_stream(tid, nthreads);

#pragma omp for schedule(static)
        for (int l = 0; l < ed; ++l)
        {
            bool output_set = false;
            for (int p = 0; p < N2; ++p)
            {
                PhantomCiphertext tmp;
                bool tmp_set = false;
                for (int j = 0; j < mid; ++j)
                {
                    for (int q = 0; q < N1; ++q)
                    {
                        std::vector<std::complex<double>> d(n, 0.0);
                        const int i = p * N1 + q;
                        for (int ind1 = 0; ind1 < (n2 / params.row_X); ++ind1)
                        {
                            for (int ind2 = 0; ind2 < params.row_X; ++ind2)
                            {
                                for (int ind3 = 0; ind3 < params.batch_size; ++ind3)
                                {
                                    const int slot = ind1 * params.row_X * params.batch_size +
                                                     ind2 * params.batch_size + ind3;
                                    const int w_row = j * n2 / params.row_X +
                                                      (i + ind1) % (n2 / params.row_X);
                                    const int w_col = l * n2 / params.row_X +
                                                      ind1 % (n2 / params.row_X);
                                    if ((j % 2) == 0)
                                    {
                                        d[slot].real(0.5 * W[w_row][w_col]);
                                    }
                                    else
                                    {
                                        d[slot].imag(-0.5 * W[w_row][w_col]);
                                    }
                                }
                            }
                        }

                        PhantomPlaintext pt_d;
                        encode_values(encoder_local, vec_left_rot_complex(d, -p * N1 * params.row_X * params.batch_size), ct_N1[j * N1 + q].params_id(), ct_N1[j * N1 + q].scale(), pt_d, stream);
                        PhantomCiphertext ct_d;
                        PhantomCiphertext ct_src = fhe_matrix_deep_copy_cipher_on_stream(ct_N1[j * N1 + q], stream);
                        evaluator_local.multiply_plain(ct_src, pt_d, ct_d, stream);
                        if (!tmp_set)
                        {
                            tmp = std::move(ct_d);
                            tmp_set = true;
                        }
                        else
                        {
                            evaluator_local.add_inplace(tmp, ct_d, stream);
                        }
                    }
                }
                evaluator_local.rotate_vector_inplace(tmp,
                                                      p * N1 * params.row_X * params.batch_size,
                                                      adapter.require_galois_keys(),
                                                      stream);
                if (!output_set)
                {
                    output.cts[l] = std::move(tmp);
                    output_set = true;
                }
                else
                {
                    evaluator_local.add_inplace(output.cts[l], tmp, stream);
                }
            }
            evaluator_local.rescale_to_next_inplace(output.cts[l], stream);
        }
        cudaStreamSynchronize(stream.get_stream());
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ext_qk_batch(FHEAdapterContext &adapter,
                                           const FHEBlockTensor &enc_QK,
                                           int batch_size,
                                           int qk_row,
                                           const std::string &out_name,
                                           const std::string &step_name = "ext_QK")
{
    
    if (batch_size <= 0 || qk_row <= 0 || enc_QK.cts.empty())
    {
        throw std::runtime_error(step_name + " requires positive dimensions and non-empty input");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int midqk = static_cast<int>(enc_QK.cts.size());
    if (qk_row % midqk != 0 || n % (qk_row * batch_size) != 0)
    {
        throw std::runtime_error(step_name + " dimensions are incompatible with ext_QK packing");
    }
    const int num_diag = qk_row / midqk;

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {qk_row, qk_row};
    output.packing = PackingType::InterleavedDiagonal;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(qk_row));

    for (int i = 0; i < midqk; ++i)
    {
        for (int j = 0; j < num_diag; ++j)
        {
            PhantomCiphertext tmp = enc_QK.cts[i];
            std::vector<double> mask(static_cast<size_t>(n), 0.0);
            for (int slot = j * qk_row * batch_size;
                 slot < (j + 1) * qk_row * batch_size;
                 ++slot)
            {
                mask[static_cast<size_t>(slot)] = 1.0;
            }

            PhantomPlaintext encoded_mask;
            encode_values(adapter, mask, tmp.params_id(), tmp.scale(), encoded_mask);
            adapter.require_evaluator().multiply_plain_inplace(tmp, encoded_mask);
            adapter.require_evaluator().rescale_to_next_inplace(tmp);

            int sum_index = n / 2;
            const int min_index = qk_row * batch_size;
            while (sum_index >= min_index)
            {
                PhantomCiphertext copy;
                adapter.require_evaluator().rotate_vector(tmp, sum_index, adapter.require_galois_keys(), copy);
                adapter.require_evaluator().add_inplace(tmp, copy);
                
                sum_index /= 2;
            }
            output.cts[static_cast<size_t>(i * num_diag + j)] = std::move(tmp);
        }
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_rot_k_for_gradient_q(FHEAdapterContext &adapter,
                                                   const FHEBlockTensor &enc_K,
                                                   const AttentionOpParams &params,
                                                   const std::string &out_name,
                                                   const std::string &step_name = "rot_K_for_gradient_Q")
{
    
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " parameters must be positive");
    }
    if (enc_K.is_complex)
    {
        throw std::runtime_error(step_name + " requires real K head");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = params.row * params.col * params.batch_size / n;
    const int ed = params.row * params.row * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        params.row * params.row * params.batch_size % n != 0 ||
        static_cast<int>(enc_K.cts.size()) != mid ||
        params.row % ed != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not match CPU rot_K expansion");
    }
    const int num_diag = params.row / ed;

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(mid * num_diag));

    for (int ind = 0; ind < mid; ++ind)
    {
        for (int diag = 0; diag < num_diag; ++diag)
        {
            PhantomCiphertext tmp = enc_K.cts[static_cast<size_t>(ind)];
            phantom_internal_rot_inplace(adapter,
                                         tmp,
                                         diag,
                                         params.row,
                                         params.batch_size,
                                         params.k,
                                         enc_K.name + "_rot_K_backward_" +
                                             std::to_string(ind) + "_" + std::to_string(diag));
            output.cts[static_cast<size_t>(ind * num_diag + diag)] = std::move(tmp);
        }
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_kernel_bsgs(FHEAdapterContext &adapter,
                                               const FHEBlockTensor &enc_Q,
                                               const FHEBlockTensor &enc_K,
                                               const AttentionOpParams &params,
                                               const std::string &out_name,
                                               const std::string &step_name = "CCMM-kernel-bsgs")
{
    
    if (params.batch_size <= 0 || params.row <= 0 || params.col <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " parameters must be positive");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = params.row * params.col * params.batch_size / n;
    const int ed = params.row * params.row * params.batch_size / n;
    if (mid <= 0 || ed <= 0 ||
        params.row * params.col * params.batch_size % n != 0 ||
        params.row * params.row * params.batch_size % n != 0 ||
        static_cast<int>(enc_Q.cts.size()) != mid ||
        static_cast<int>(enc_K.cts.size()) != mid ||
        params.row % ed != 0)
    {
        throw std::runtime_error(step_name + " dimensions do not match kernel BSGS packing");
    }

    FHEBlockTensor Q = enc_Q;
    FHEBlockTensor K = enc_K;
    const int start_level = std::min(Q.level, K.level);
    align_down_to(adapter.require_context(), adapter.level_map, Q, start_level, Q.name);
    align_down_to(adapter.require_context(), adapter.level_map, K, start_level, K.name);
    if (start_level < 3)
    {
        
        throw std::runtime_error(step_name + " requires at least three remaining levels");
    }

    const int num_diag = params.row / ed;

    std::vector<PhantomCiphertext> rot_Q(static_cast<size_t>(ed * mid));
    for (int ind = 0; ind < ed; ++ind)
    {
        for (int j = 0; j < mid; ++j)
        {
            rot_Q[static_cast<size_t>(ind * mid + j)] = Q.cts[j];
            phantom_internal_rot_inplace(adapter,
                                         rot_Q[static_cast<size_t>(ind * mid + j)],
                                         params.row - (ind * params.row / ed),
                                         params.row,
                                         params.batch_size,
                                         params.k,
                                         Q.name + "_rot_Q_" + std::to_string(ind) + "_" + std::to_string(j));
        }
    }

    std::vector<PhantomCiphertext> rot_K(static_cast<size_t>(num_diag * mid));
    for (int i = 0; i < num_diag; ++i)
    {
        for (int j = 0; j < mid; ++j)
        {
            rot_K[static_cast<size_t>(i * mid + j)] = K.cts[j];
            phantom_internal_rot_inplace(adapter,
                                         rot_K[static_cast<size_t>(i * mid + j)],
                                         i,
                                         params.row,
                                         params.batch_size,
                                         params.k,
                                         K.name + "_rot_K_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row, params.row};
    output.packing = PackingType::InterleavedDiagonal;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(ed));

    for (int ind = 0; ind < ed; ++ind)
    {
        bool output_set = false;
        for (int i = 0; i < num_diag; ++i)
        {
            PhantomCiphertext tmpoutput;
            bool tmpoutput_set = false;
            for (int j = 0; j < mid; ++j)
            {
                PhantomCiphertext lhs = rot_Q[static_cast<size_t>(ind * mid + j)];
                PhantomCiphertext rhs = rot_K[static_cast<size_t>(i * mid + j)];
                assert_same_level_for_add(adapter.level_map,
                                          lhs,
                                          rhs,
                                          Q.name + "_rot_Q",
                                          K.name + "_rot_K");
                PhantomCiphertext tmpQK;
                adapter.require_evaluator().sub(lhs, rhs, tmpQK);
                adapter.require_evaluator().square_inplace(tmpQK);
                
                if (!tmpoutput_set)
                {
                    tmpoutput = std::move(tmpQK);
                    tmpoutput_set = true;
                }
                else
                {
                    adapter.require_evaluator().add_inplace(tmpoutput, tmpQK);
                    
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
                adapter.require_evaluator().add_inplace(tmpoutput, copy);
                
                sum_index /= 2;
            }

            std::vector<double> mask(static_cast<size_t>(n), 0.0);
            for (int slot = i * params.row * params.batch_size;
                 slot < (i + 1) * params.row * params.batch_size;
                 ++slot)
            {
                mask[static_cast<size_t>(slot)] = params.distance_scale;
            }
            PhantomPlaintext encoded_mask;
            encode_values(adapter, mask, tmpoutput.params_id(), tmpoutput.scale(), encoded_mask);
            adapter.require_evaluator().multiply_plain_inplace(tmpoutput, encoded_mask);
            adapter.require_evaluator().rescale_to_next_inplace(tmpoutput);

            if (!output_set)
            {
                output.cts[static_cast<size_t>(ind)] = std::move(tmpoutput);
                output_set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(ind)], tmpoutput);
                
            }
        }
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_lora_a(FHEAdapterContext &adapter,
                                          const FHEBlockTensor &enc_X,
                                          const FHEBlockTensor &W,
                                          const MatrixOpParams &params,
                                          const std::string &out_name,
                                          const std::string &step_name = "CCMM-A")
{
    
    validate_matrix_params(params);

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = params.row_X * params.col_X * params.batch_size / n;
    const int ed = params.col_W;
    if (mid <= 0 || static_cast<int>(enc_X.cts.size()) != mid ||
        static_cast<int>(W.cts.size()) != mid * params.col_W)
    {
        throw std::runtime_error(step_name + " ciphertext count mismatch");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::RepeatedColumn;
    output.is_complex = false;
    output.cts.resize(ed);

    for (int i = 0; i < params.col_W; ++i)
    {
        bool set = false;
        for (int j = 0; j < mid; ++j)
        {
            PhantomCiphertext lhs = enc_X.cts[j];
            PhantomCiphertext rhs = W.cts[i * mid + j];
            align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, enc_X.name + "/ct_" + std::to_string(j), W.name + "/ct_" + std::to_string(i * mid + j));
            PhantomCiphertext xa;
            adapter.require_evaluator().multiply(lhs, rhs, xa);
            
            if (!set)
            {
                output.cts[i] = std::move(xa);
                set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(output.cts[i], xa);
                
            }
        }
        adapter.require_evaluator().relinearize_inplace(output.cts[i], adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(output.cts[i]);

        int rot_index = n / 2;
        while (rot_index >= params.row_X * params.batch_size)
        {
            PhantomCiphertext tmp;
            adapter.require_evaluator().rotate_vector(output.cts[i], rot_index, adapter.require_galois_keys(), tmp);
            adapter.require_evaluator().add_inplace(output.cts[i], tmp);
            
            rot_index /= 2;
        }
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_lora_b_complex(FHEAdapterContext &adapter,
                                                  const FHEBlockTensor &enc_X,
                                                  const FHEBlockTensor &W,
                                                  const MatrixOpParams &params,
                                                  const std::string &out_name,
                                                  const std::string &step_name = "CCMM-B-complex")
{
    
    validate_matrix_params(params);

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int mid = static_cast<int>(enc_X.cts.size());
    const int ed = params.row_X * params.col_W * params.batch_size / (2 * n);
    if (ed <= 0 || params.row_X * params.col_W * params.batch_size % (2 * n) != 0 ||
        static_cast<int>(W.cts.size()) != mid * ed)
    {
        throw std::runtime_error(step_name + " ciphertext count mismatch");
    }

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.row_X, params.col_W};
    output.packing = PackingType::InterleavedColumnComplex;
    output.is_complex = true;
    output.cts.resize(ed);

    for (int i = 0; i < ed; ++i)
    {
        bool set = false;
        for (int j = 0; j < mid; ++j)
        {
            PhantomCiphertext lhs = enc_X.cts[j];
            PhantomCiphertext rhs = W.cts[j * ed + i];
            align_pair_for_mul(adapter.require_context(), adapter.level_map, lhs, rhs, enc_X.name + "/ct_" + std::to_string(j), W.name + "/ct_" + std::to_string(j * ed + i));
            PhantomCiphertext tmp;
            adapter.require_evaluator().multiply(lhs, rhs, tmp);
            
            if (!set)
            {
                output.cts[i] = std::move(tmp);
                set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(output.cts[i], tmp);
                
            }
        }
        adapter.require_evaluator().relinearize_inplace(output.cts[i], adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(output.cts[i]);
        
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_v_bsgs_complex_from_ext(FHEAdapterContext &adapter,
                                                           const FHEBlockTensor &ext_QK,
                                                           const FHEBlockTensor &enc_V,
                                                           const AttentionValueOpParams &params,
                                                           const std::string &out_name,
                                                           const std::string &step_name = "CCMM-V-bsgs-complex/from_ext")
{
    
    if (params.batch_size <= 0 || params.qk_row <= 0 || params.v_col <= 0 ||
        params.v_row <= 0 || params.k <= 0)
    {
        throw std::runtime_error(step_name + " parameters must be positive");
    }

    const int n = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    const int midv = static_cast<int>(enc_V.cts.size());
    const int midqk = midv * 4;
    const int ed = params.qk_row * params.v_col * params.batch_size / n;
    if (midv <= 0 || ed <= 0 || ed != 2 * midv ||
        params.qk_row * params.v_col * params.batch_size % n != 0 ||
        params.qk_row % midqk != 0 ||
        static_cast<int>(ext_QK.cts.size()) != params.qk_row)
    {
        throw std::runtime_error(step_name + " dimensions do not match CCMM-V BSGS complex packing");
    }
    const int num_diag = params.qk_row / midqk;

    FHEBlockTensor output;
    output.name = out_name;
    output.logical_shape = {params.v_row, params.v_col};
    output.packing = PackingType::InterleavedColumnMix;
    output.is_complex = true;
    output.cts.resize(static_cast<size_t>(ed));

    for (int ind = 0; ind < midv; ++ind)
    {
        std::vector<PhantomCiphertext> tmpv(static_cast<size_t>(num_diag), enc_V.cts[static_cast<size_t>(ind)]);
        for (int j = 0; j < num_diag; ++j)
        {
            phantom_internal_rot_inplace(adapter,
                                         tmpv[static_cast<size_t>(j)],
                                         j,
                                         params.v_row,
                                         params.batch_size,
                                         params.k,
                                         enc_V.name + "_rot_V_" + std::to_string(ind) + "_" + std::to_string(j));
        }

        bool even_set = false;
        for (int i = 0; i < midqk; ++i)
        {
            PhantomCiphertext tmpqkv;
            bool tmpqkv_set = false;
            for (int j = 0; j < num_diag; ++j)
            {
                PhantomCiphertext tmp = ext_QK.cts[static_cast<size_t>(i * num_diag + j)];
                PhantomCiphertext rhs = tmpv[static_cast<size_t>(j)];
                align_pair_for_mul(adapter.require_context(), adapter.level_map, tmp, rhs, ext_QK.name + "/ct_" + std::to_string(i * num_diag + j), enc_V.name + "_rot_V_" + std::to_string(j));
                adapter.require_evaluator().multiply_inplace(tmp, rhs);
                
                if (!tmpqkv_set)
                {
                    tmpqkv = std::move(tmp);
                    tmpqkv_set = true;
                }
                else
                {
                    adapter.require_evaluator().add_inplace(tmpqkv, tmp);
                    
                }
            }

            adapter.require_evaluator().relinearize_inplace(tmpqkv, adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(tmpqkv);
            
            phantom_internal_rot_inplace(adapter,
                                         tmpqkv,
                                         i * params.qk_row / midqk,
                                         params.qk_row,
                                         params.batch_size,
                                         params.k,
                                         out_name + "_accum_rot_" + std::to_string(ind) + "_" + std::to_string(i));

            if (!even_set)
            {
                output.cts[static_cast<size_t>(2 * ind)] = std::move(tmpqkv);
                even_set = true;
            }
            else
            {
                adapter.require_evaluator().add_inplace(output.cts[static_cast<size_t>(2 * ind)], tmpqkv);
                
            }
        }

        adapter.require_evaluator().complex_conjugate(output.cts[static_cast<size_t>(2 * ind)],
                                                      adapter.require_galois_keys(),
                                                      output.cts[static_cast<size_t>(2 * ind + 1)]);
        PhantomCiphertext sum;
        adapter.require_evaluator().add(output.cts[static_cast<size_t>(2 * ind)],
                                        output.cts[static_cast<size_t>(2 * ind + 1)],
                                        sum);
        adapter.require_evaluator().sub(output.cts[static_cast<size_t>(2 * ind)],
                                        output.cts[static_cast<size_t>(2 * ind + 1)],
                                        output.cts[static_cast<size_t>(2 * ind + 1)]);
        output.cts[static_cast<size_t>(2 * ind)] = std::move(sum);
        
    }

    publish_matrix_output(output);
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);
    
    return output;
}

inline FHEBlockTensor phantom_ccmm_v_bsgs_complex(FHEAdapterContext &adapter,
                                                  const FHEBlockTensor &enc_QK,
                                                  const FHEBlockTensor &enc_V,
                                                  const AttentionValueOpParams &params,
                                                  const std::string &out_name,
                                                  FHEBlockTensor *ext_cache = nullptr,
                                                  const std::string &step_name = "CCMM-V-bsgs-complex")
{

    FHEBlockTensor qk = enc_QK;
    FHEBlockTensor v = enc_V;
    const int start_level = std::min(qk.level, v.level);
    align_down_to(adapter.require_context(), adapter.level_map, qk, start_level, qk.name);
    align_down_to(adapter.require_context(), adapter.level_map, v, start_level, v.name);
    if (start_level < 3)
    {
        
        throw std::runtime_error(step_name + " requires at least three remaining levels");
    }

    FHEBlockTensor ext = phantom_ext_qk_batch(adapter,
                                              qk,
                                              params.batch_size,
                                              params.qk_row,
                                              out_name + "_ext_QK",
                                              step_name + "/ext_QK");
    if (ext_cache != nullptr)
    {
        *ext_cache = ext;
    }
    return phantom_ccmm_v_bsgs_complex_from_ext(adapter,
                                                ext,
                                                v,
                                                params,
                                                out_name,
                                                step_name + "/matmul");
}

}
