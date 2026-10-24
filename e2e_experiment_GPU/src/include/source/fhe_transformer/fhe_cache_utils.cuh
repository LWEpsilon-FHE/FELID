#pragma once

#include <cstdlib>
#include "util/execution_timer.h"
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "source/fhe_transformer/fhe_types.cuh"
#include "source/fhe_transformer/fhe_level_utils.cuh"
#include "source/fhe_transformer/fhe_types.cuh"

namespace fhe_transformer {

inline void offload_tensor_to_host(FHEBlockTensor &tensor,
                                   FHEHostTensorPayload &payload)
{
    if (payload.offloaded || tensor.cts.empty())
    {
        return;
    }
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    payload.ciphertext_words.clear();
    payload.streams.clear();
    payload.ciphertext_words.resize(tensor.cts.size());
    payload.streams.resize(tensor.cts.size());
    for (std::size_t i = 0; i < tensor.cts.size(); ++i)
    {
        PhantomCiphertext &ct = tensor.cts[i];
        const std::size_t words = ct.size() * ct.coeff_modulus_size() * ct.poly_modulus_degree();
        payload.streams[i] = ct.data_ptr().get_stream();
        payload.ciphertext_words[i].resize(words);
        if (words != 0)
        {

            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(payload.ciphertext_words[i].data(),
                                          ct.data(),
                                          words * sizeof(std::uint64_t),
                                          cudaMemcpyDeviceToHost,
                                          payload.streams[i]));
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(payload.streams[i]));
            ct.data_ptr().reset();
        }
    }
    payload.offloaded = true;
}

inline void restore_tensor_from_host(FHEBlockTensor &tensor,
                                     FHEHostTensorPayload &payload)
{
    if (!payload.offloaded)
    {
        return;
    }
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    if (payload.ciphertext_words.size() != tensor.cts.size())
    {
        throw std::runtime_error("host-offload ciphertext-count mismatch for " + tensor.name);
    }
    for (std::size_t i = 0; i < tensor.cts.size(); ++i)
    {
        const std::size_t words = payload.ciphertext_words[i].size();
        if (words == 0)
        {
            continue;
        }
        cudaStream_t stream = payload.streams[i];
        auto device_data = phantom::util::make_cuda_auto_ptr<std::uint64_t>(words, stream);
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(device_data.get(),
                                           payload.ciphertext_words[i].data(),
                                           words * sizeof(std::uint64_t),
                                           cudaMemcpyHostToDevice,
                                           stream));
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));
        tensor.cts[i].data_ptr() = std::move(device_data);
    }
    payload = FHEHostTensorPayload{};
}

inline void cache_write(std::unordered_map<std::string, FHECacheRecord> &records, const std::string &key, FHEBlockTensor tensor, const std::string &why_cached, const std::string &consumer)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    tensor.is_cached = true;
    tensor.why_cached = why_cached;
    tensor.consumer = consumer;
    tensor.backward_consumer = consumer;
    records[key] = FHECacheRecord{tensor, why_cached, consumer, {}};

    offload_tensor_to_host(records.at(key).tensor, records.at(key).host_payload);
}

inline const FHEBlockTensor &cache_read(const std::unordered_map<std::string, FHECacheRecord> &records, const std::string &key, const std::string &consumer, int required_level = -1)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    const auto it = records.find(key);
    if (it == records.end())
    {
        throw std::runtime_error("[CACHE ERROR] missing cache tensor name=" + key + " consumer=" + consumer);
    }
    FHECacheRecord &record = const_cast<FHECacheRecord &>(it->second);
    restore_tensor_from_host(record.tensor, record.host_payload);
    const FHEBlockTensor &tensor = record.tensor;
    if (required_level >= 0 && tensor.level != required_level)
    {
        std::ostringstream oss;
        oss << "[CACHE ERROR] " << key << " level=" << tensor.level
            << " required=" << required_level
            << " consumer=" << consumer
            << ". No silent replacement or recomputation is allowed.";
        throw std::runtime_error(oss.str());
    }

    return tensor;
}

inline FHEBlockTensor cache_take(std::unordered_map<std::string, FHECacheRecord> &records, const std::string &key, const std::string &consumer, int required_level = -1)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    auto it = records.find(key);
    if (it == records.end())
    {
        throw std::runtime_error("[CACHE ERROR] missing cache tensor name=" + key + " consumer=" + consumer);
    }
    restore_tensor_from_host(it->second.tensor, it->second.host_payload);
    FHEBlockTensor tensor = std::move(it->second.tensor);
    if (required_level >= 0 && tensor.level != required_level)
    {
        std::ostringstream oss;
        oss << "[CACHE ERROR] " << key << " level=" << tensor.level
            << " required=" << required_level
            << " consumer=" << consumer
            << ". No silent replacement or recomputation is allowed.";
        throw std::runtime_error(oss.str());
    }

    records.erase(it);
    
    return tensor;
}

inline void cache_write(FHEForwardCache &cache, const std::string &key, const FHEBlockTensor &tensor, const std::string &why_cached, const std::string &consumer)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    cache_write(cache.records, key, tensor, why_cached, consumer);
}

inline const FHEBlockTensor &cache_read(const FHEForwardCache &cache, const std::string &key, const std::string &consumer, int required_level = -1)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    return cache_read(cache.records, key, consumer, required_level);
}

inline FHEBlockTensor cache_take(FHEForwardCache &cache, const std::string &key, const std::string &consumer, int required_level = -1)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    return cache_take(cache.records, key, consumer, required_level);
}

inline void cache_write(FHEBackwardCache &cache, const std::string &key, const FHEBlockTensor &tensor, const std::string &why_cached, const std::string &consumer)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    cache_write(cache.records, key, tensor, why_cached, consumer);
}

inline const FHEBlockTensor &cache_read(const FHEBackwardCache &cache, const std::string &key, const std::string &consumer, int required_level = -1)
{
    phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
    return cache_read(cache.records, key, consumer, required_level);
}

}
