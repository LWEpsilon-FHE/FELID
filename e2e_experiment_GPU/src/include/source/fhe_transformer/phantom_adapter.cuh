#pragma once

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "phantom.h"
#include "source/ckks_evaluator_parallel.cuh"
#include "source/fhe_transformer/fhe_level_utils.cuh"

namespace fhe_transformer {

struct FHEAdapterContext {
    PhantomContext *context = nullptr;
    PhantomCKKSEncoder *encoder = nullptr;
    moai::Evaluator *evaluator = nullptr;
    moai::Encoder *ckks_encoder = nullptr;
    PhantomRelinKey *relin_keys = nullptr;
    PhantomGaloisKey *galois_keys = nullptr;
    FHELevelMap level_map;

    const PhantomContext &require_context() const
    {
        if (context == nullptr)
        {
            throw std::runtime_error("FHEAdapterContext.context is null");
        }
        return *context;
    }

    moai::Evaluator &require_evaluator() const
    {
        if (evaluator == nullptr)
        {
            throw std::runtime_error("FHEAdapterContext.evaluator is null");
        }
        return *evaluator;
    }

    moai::Encoder &require_ckks_encoder() const
    {
        if (ckks_encoder == nullptr)
        {
            throw std::runtime_error("FHEAdapterContext.ckks_encoder is null");
        }
        return *ckks_encoder;
    }

    PhantomRelinKey &require_relin_keys() const
    {
        if (relin_keys == nullptr)
        {
            throw std::runtime_error("FHEAdapterContext.relin_keys is null");
        }
        return *relin_keys;
    }

    PhantomGaloisKey &require_galois_keys() const
    {
        if (galois_keys == nullptr)
        {
            throw std::runtime_error("FHEAdapterContext.galois_keys is null");
        }
        return *galois_keys;
    }
};

template <typename ValuesT>
inline void encode_values(FHEAdapterContext &adapter,
                              ValuesT &&values,
                              size_t chain_index,
                              double scale,
                              PhantomPlaintext &plain)
{
    adapter.require_ckks_encoder().encode(std::forward<ValuesT>(values), chain_index, scale, plain);
}

template <typename ValuesT>
inline void encode_values(moai::Encoder &encoder, ValuesT &&values, size_t chain_index, double scale, PhantomPlaintext &plain)
{
    encoder.encode(std::forward<ValuesT>(values), chain_index, scale, plain);
}

template <typename ValuesT>
inline void encode_values(moai::Encoder &encoder, ValuesT &&values, size_t chain_index, double scale, PhantomPlaintext &plain, const phantom::util::cuda_stream_wrapper &stream)
{
    encoder.encode(std::forward<ValuesT>(values), chain_index, scale, plain, stream);
}

using CipherUnaryCallback = std::function<void(PhantomCiphertext &out, const PhantomCiphertext &in)>;
using CipherPairCallback = std::function<void(PhantomCiphertext &out0,
                                              PhantomCiphertext &out1,
                                              const PhantomCiphertext &in0,
                                              const PhantomCiphertext &in1)>;
using GeluPostprocessCallback = std::function<FHEBlockTensor(FHEAdapterContext &adapter,
                                                             const FHEBlockTensor &gelu_input,
                                                             const FHEBlockTensor &gelu_derivative,
                                                             int stored_level,
                                                             const std::string &out_name,
                                                             const std::string &step_name)>;
using GeluInputCacheCallback = std::function<FHEBlockTensor(FHEAdapterContext &adapter,
                                                            const FHEBlockTensor &gelu_input,
                                                            int stored_level,
                                                            const std::string &out_name,
                                                            const std::string &step_name)>;
using GeluTanhComponentCallback = std::function<FHEBlockTensor(FHEAdapterContext &adapter,
                                                               const FHEBlockTensor &gelu_input,
                                                               int stored_level,
                                                               const std::string &out_name,
                                                               const std::string &step_name)>;

struct PhantomBootstrappingCallbacks {
    CipherUnaryCallback ordinary_bts;

    std::function<CipherUnaryCallback(double)> gain_compensated_bts;
    CipherUnaryCallback rboot_tanh;
    CipherUnaryCallback rboot_relu;
    CipherUnaryCallback exp_fb;
    CipherPairCallback exp_fb_pair;
    GeluPostprocessCallback gelu_output_component;
    GeluInputCacheCallback gelu_input_for_backward;
    GeluTanhComponentCallback gelu_derivative_component;
};

}
