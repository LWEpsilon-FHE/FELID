#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

#include "phantom.h"
#include "source/ckks_evaluator_parallel.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace fhe;

using phantom::util::cuda_stream_wrapper;
using std::vector;

inline vector<cuda_stream_wrapper> stream_pool;

static inline std::size_t ciphertext_device_bytes(const PhantomCiphertext &ct)
{
    if (ct.data() == nullptr)
    {
        return 0;
    }
    return ct.size() * ct.coeff_modulus_size() * ct.poly_modulus_degree() * sizeof(uint64_t);
}

static inline std::size_t output_device_bytes(const std::vector<PhantomCiphertext> &output)
{
    std::size_t total = 0;
    for (const auto &ct : output)
    {
        total += ciphertext_device_bytes(ct);
    }
    return total;
}

static inline void print_output_mem(const std::vector<PhantomCiphertext> &output)
{
    const double mb = 1024.0 * 1024.0;
    std::cout << "[GPU] output device memory ~ " << (output_device_bytes(output) / mb) << " MB" << std::endl;
}

static inline void print_output_mem(const std::vector<std::vector<PhantomCiphertext>> &output)
{
    const double mb = 1024.0 * 1024.0;
    std::size_t total = 0;
    for (const auto &vec_ct : output)
    {
        total += output_device_bytes(vec_ct);
    }
    std::cout << "[GPU] output device memory ~ " << (total / mb) << " MB" << std::endl;
}

static inline void print_cuda_meminfo(const char *tag)
{
    size_t free_b = 0;
    size_t total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    const double mb = 1024.0 * 1024.0;
    std::cout << tag << " Free: " << free_b / mb << " MB, Total: " << total_b / mb << " MB\n";
}

static inline void bridge_to_default(const cuda_stream_wrapper &sw)
{
    auto dst = phantom::util::global_variables::default_stream->get_stream();
    if (sw.get_stream() == dst)
    {
        return;
    }

    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, sw.get_stream());
    cudaStreamWaitEvent(dst, ev, 0);
    cudaEventDestroy(ev);
}

inline PhantomCiphertext deep_copy_cipher(
    const PhantomCiphertext &src,
    const PhantomContext &context,
    phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    (void)context;

    PhantomCiphertext dst;
    dst.set_chain_index(src.chain_index());
    dst.set_poly_modulus_degree(src.poly_modulus_degree());
    dst.set_coeff_modulus_size(src.coeff_modulus_size());
    dst.set_scale(src.scale());
    dst.set_correction_factor(src.correction_factor());
    dst.set_ntt_form(src.is_ntt_form());
    dst.SetNoiseScaleDeg(src.GetNoiseScaleDeg());

    dst.reinit_like(src.size(), src.coeff_modulus_size(), src.poly_modulus_degree(), stream.get_stream());

    const size_t count = src.size() * src.coeff_modulus_size() * src.poly_modulus_degree();
    cudaMemcpyAsync(dst.data(), src.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream.get_stream());

    return dst;
}

inline vector<PhantomCiphertext> deep_copy_cipher(
    const vector<PhantomCiphertext> &src,
    const PhantomContext &context,
    phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    (void)context;

    vector<PhantomCiphertext> dst(src.size());
    for (size_t idx = 0; idx < src.size(); idx++)
    {
        dst[idx].set_chain_index(src[idx].chain_index());
        dst[idx].set_poly_modulus_degree(src[idx].poly_modulus_degree());
        dst[idx].set_coeff_modulus_size(src[idx].coeff_modulus_size());
        dst[idx].set_scale(src[idx].scale());
        dst[idx].set_correction_factor(src[idx].correction_factor());
        dst[idx].set_ntt_form(src[idx].is_ntt_form());
        dst[idx].SetNoiseScaleDeg(src[idx].GetNoiseScaleDeg());

        dst[idx].reinit_like(
            src[idx].size(),
            src[idx].coeff_modulus_size(),
            src[idx].poly_modulus_degree(),
            stream.get_stream());

        const size_t count = src[idx].size() * src[idx].coeff_modulus_size() * src[idx].poly_modulus_degree();
        cudaMemcpyAsync(
            dst[idx].data(),
            src[idx].data(),
            count * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            stream.get_stream());
    }

    return dst;
}

inline PhantomPlaintext deep_copy_plain(
    const PhantomPlaintext &src,
    const PhantomContext &context,
    phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    (void)context;

    PhantomPlaintext dst;
    dst.set_parms_id(src.parms_id());
    dst.set_chain_index(src.chain_index());
    dst.set_poly_modulus_degree(src.poly_modulus_degree());
    dst.set_coeff_modulus_size(src.coeff_modulus_size());
    dst.set_scale(src.scale());

    dst.reinit_like(src.coeff_modulus_size(), src.poly_modulus_degree(), stream.get_stream());

    const size_t count = src.coeff_modulus_size() * src.poly_modulus_degree();
    cudaMemcpyAsync(dst.data(), src.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream.get_stream());

    return dst;
}

inline vector<PhantomPlaintext> deep_copy_plain(
    const vector<PhantomPlaintext> &src,
    const PhantomContext &context,
    phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream)
{
    (void)context;

    vector<PhantomPlaintext> dst(src.size());
    for (size_t idx = 0; idx < src.size(); idx++)
    {
        dst[idx].set_parms_id(src[idx].parms_id());
        dst[idx].set_chain_index(src[idx].chain_index());
        dst[idx].set_poly_modulus_degree(src[idx].poly_modulus_degree());
        dst[idx].set_coeff_modulus_size(src[idx].coeff_modulus_size());
        dst[idx].set_scale(src[idx].scale());

        dst[idx].reinit_like(src[idx].coeff_modulus_size(), src[idx].poly_modulus_degree(), stream.get_stream());

        const size_t count = src[idx].coeff_modulus_size() * src[idx].poly_modulus_degree();
        cudaMemcpyAsync(
            dst[idx].data(),
            src[idx].data(),
            count * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            stream.get_stream());
    }

    return dst;
}

inline const complex<double> coeff25_i(0.0, 1.0);
inline const vector<complex<double>> coeff25{
    0.18062800362446218561 / 2.0 + 8.4301680513910193717e-16 / 2.0 * coeff25_i,
    -2.6814708645607593834e-17 + 0.18179610866713943884 * coeff25_i,
    0.17136920383910317356 + 1.2419444004281411103e-16 * coeff25_i,
    -3.3224364878877831627e-17 + 0.1992516324333585831 * coeff25_i,
    0.1409257969070405736 + -6.0215486081364422323e-17 * coeff25_i,
    -1.7123778854388006404e-16 + 0.2279608000326154571 * coeff25_i,
    0.082876055856842240077 + -4.2903533832972150134e-16 * coeff25_i,
    5.2035823860741581761e-16 + 0.25328585722348251341 * coeff25_i,
    -0.0074221436141012411825 + 3.4247557708776012808e-16 * coeff25_i,
    2.8161324496450605368e-16 + 0.25026180386150653767 * coeff25_i,
    -0.12213370469086201608 + 6.6613381477509392425e-16 * coeff25_i,
    -1.646517197537308471e-18 + 0.18805961883854194205 * coeff25_i,
    -0.22748947981900527471 + 2.5215234796571353773e-16 * coeff25_i,
    3.218353079329174737e-16 + 0.049028290014482729664 * coeff25_i,
    -0.2599503538007409964 + -3.9516412740895400531e-16 * coeff25_i,
    1.8358666752540989982e-16 + -0.13631998925663849076 * coeff25_i,
    -0.15580955316508710018 + 2.182811370449460417e-16 * coeff25_i,
    -1.275933219720161277e-15 + -0.26328503536051234279 * coeff25_i,
    0.072143391454352293057 + 5.2688550321193871073e-17 * coeff25_i,
    -1.2836953722228372499e-16 + -0.19714884575899777053 * coeff25_i,
    0.26291684848498231286 + -6.6613381477509392425e-16 * coeff25_i,
    5.2335725207435874885e-16 + 0.070656057015580170377 * coeff25_i,
    0.18734869635645157171 + 5.1559509957168287461e-16 * coeff25_i,
    8.4207593816908057127e-16 + 0.28057105360851963827 * coeff25_i,
    -0.14130673136093591102 + 4.8925082441108591134e-16 * coeff25_i,
    6.7924714816976604587e-16 + 0.10785042803474780004 * coeff25_i,
    -0.27862616125139233469 + 1.3924831156315523078e-15 * coeff25_i,
    3.5447163095553195164e-16 + -0.26109773253640045088 * coeff25_i,
    0.080408993503119960411 + 2.2580807280511657447e-16 * coeff25_i,
    7.2352669994639438447e-16 + -0.14643223302221275439 * coeff25_i,
    0.29668323276411645573 + 1.038717134903536282e-15 * coeff25_i,
    -4.0616051012107820157e-16 + 0.3068663560359553566 * coeff25_i,
    -0.18780259775854354909 + -1.0914056852247302085e-16 * coeff25_i,
    8.2155327738549053833e-16 + 0.00079570762613815216712 * coeff25_i,
    -0.18913992462719783627 + 8.5430720877935773631e-16 * coeff25_i,
    -2.6250188463594802028e-16 + -0.32672007924592583183 * coeff25_i,
    0.39325017030968856258 + -1.1742019785866062661e-15 * coeff25_i,
    5.4590865726205724107e-16 + 0.39429032240354472405 * coeff25_i,
    -0.34974835496435485727 + -2.1865748383295454862e-15 * coeff25_i,
    1.4363510381087962742e-15 + -0.28258610069142042764 * coeff25_i,
    0.21153933021645898727 + 9.6909297912195875534e-16 * coeff25_i,
    -8.4795635673171386162e-17 + 0.14835828410599483096 * coeff25_i,
    -0.098249509728547942955 + 7.5269357601705524822e-16 * coeff25_i,
    -9.95937089860379409e-16 + -0.061801586436542474412 * coeff25_i,
    0.037094235170279959979 + 3.6129291648818655859e-16 * coeff25_i,
    3.0772230338259767547e-16 + 0.021322944460262581445 * coeff25_i,
    -0.011774353804613129151 + -7.9973692451812127519e-16 * coeff25_i,
    -1.019194145275593827e-15 + -0.0062615496337558784398 * coeff25_i,
    0.0032138570962525446484 + 1.4978602162739399637e-15 * coeff25_i,
    5.9362825389782596455e-17 + 0.0015951094513298280479 * coeff25_i,
    -0.00076681754685429799698 + 1.6897970781582891407e-15 * coeff25_i,
    -7.4178539958337061665e-16 + -0.00035757527803138584171 * coeff25_i,
    0.00016195195640400752621 + 5.6546104898281280206e-16 * coeff25_i,
    1.1257326285841017359e-15 + 7.1327121189517490407e-05 * coeff25_i,
    -3.0582578261238026294e-05 + 8.8629668576008259915e-16 * coeff25_i,
    1.0524405617240816361e-15 + -1.2770480552385539465e-05 * coeff25_i,
    5.2199382988430785755e-06 + -5.6922451686289806844e-16 * coeff25_i,
    6.4813238345023286309e-17 + 2.028849383161462318e-06 * coeff25_i,
    -9.1760095736556924413e-07 + 1.0884654759434134894e-15 * coeff25_i};

inline constexpr size_t SMALL_POLY_DEGREE = 16384;

#include "source/bootstrapping/Bootstrapper.cuh"
#include "source/utils.cuh"
#include "source/utils_fhe.cuh"

// HE-Finetune source
#include "source/he_finetune/Packing.cuh"
#include "source/he_finetune/PCMM.cuh"
#include "source/he_finetune/CCMM_classifier.cuh"
#include "source/he_finetune/CCMM.cuh"
#include "source/he_finetune/CCMM_back.cuh"
#include "source/he_finetune/Layernorm.cuh"
#include "source/he_finetune/Layernorm_v2.cuh"
#include "source/he_finetune/Softmax_classifier.cuh"
#include "source/he_finetune/back_ffn.cuh"
#include "source/he_finetune/back_att.cuh"
#include "source/he_finetune/back_classifier.cuh"
#include "source/he_finetune/back_lora.cuh"

#include "source/ModRaise.cuh"
#include "source/PolyEval_2.cuh"
#include "source/PolyEval.cuh"
#include "source/FuncBoot_v2.cuh"
#include "source/FuncBoot_v2_out1.cuh"
#include "source/FuncBoot_v2_2ct.cuh"
#include "source/FuncBoot_v2_2ct_parallel.cuh"

// Current test entry points
#include "test/test_phantom_ckks.cuh"
#include "test/bootstrapping/bootstrapping.cuh"
#include "test/he_finetune/test_PCMM.cuh"
#include "test/he_finetune/test_CCMM_kernel.cuh"
#include "test/he_finetune/test_CCMM_LoraAB.cuh"
#include "test/he_finetune/test_CCMM_V.cuh"
#include "test/he_finetune/test_CCMM_class.cuh"
#include "test/he_finetune/test_kernelV_bsgs_selfoutput.cuh"
#include "test/he_finetune/test_layernorm.cuh"
#include "test/he_finetune/test_packing.cuh"
#include "test/he_finetune/test_softmax_class.cuh"
#include "test/functional_boostrapping/test_ckks_funcboot_v2.cuh"
#include "test/functional_boostrapping/test_ckks_funcboot_v2_out1.cuh"
#include "test/functional_boostrapping/test_ckks_funcboot_v2_2ct.cuh"
#include "test/functional_boostrapping/test_ckks_funcboot_v2_2ct_parallel.cuh"

#endif
