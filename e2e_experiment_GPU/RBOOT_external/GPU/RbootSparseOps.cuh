#pragma once

#include <complex>
#include <mutex>
#include <vector>

#include "source/bootstrapping/Bootstrapper.cuh"
#include "source/ckks_evaluator_parallel.cuh"

namespace rboot_slim {

class RbootSparseOps {
public:
    moai::CKKSEvaluator *ckks = nullptr;
    PhantomCKKSEncoder *raw_encoder = nullptr;
    size_t configured_sparse_slots = 0;
    mutable std::mutex encode_mutex;

    RbootSparseOps() = default;
    RbootSparseOps(moai::CKKSEvaluator *ckks,
                   PhantomCKKSEncoder *raw_encoder,
                   size_t configured_sparse_slots = 0);

    size_t full_slot_count() const;
    size_t sparse_slot_count() const;

    template <typename T>
    void encode_exact_vector(const std::vector<T> &values,
                             double scale,
                             PhantomPlaintext &plain,
                             size_t chain_index = 1,
                             const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const
    {
        if (raw_encoder == nullptr)
        {
            throw std::invalid_argument("RbootSparseOps requires a valid raw encoder.");
        }
        const std::lock_guard<std::mutex> lock(encode_mutex);
        raw_encoder->reset_sparse_slots();
        raw_encoder->encode(*(ckks->context), values, scale, plain, chain_index, stream_wrapper);
    }

    template <typename T>
    void multiply_vector_reduced_error(const PhantomCiphertext &ct,
                                       const std::vector<T> &values,
                                       PhantomCiphertext &dest,
                                       const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const
    {
        dest = ct;
        multiply_vector_inplace_reduced_error(dest, values, stream_wrapper);
    }

    template <typename T>
    void multiply_vector_inplace_reduced_error(PhantomCiphertext &ct,
                                               const std::vector<T> &values,
                                               const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const
    {
        PhantomPlaintext plain;
        encode_exact_vector(values, ct.scale(), plain, 1, stream_wrapper);
        ckks->evaluator.mod_switch_to_inplace(plain, ct.params_id(), stream_wrapper);
        ckks->evaluator.multiply_plain_inplace(ct, plain, stream_wrapper);
    }

    void multiply_const(const PhantomCiphertext &ct,
                        double value,
                        PhantomCiphertext &dest,
                        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
    void multiply_const_inplace(PhantomCiphertext &ct,
                                double value,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void add_const(const PhantomCiphertext &ct,
                   double value,
                   PhantomCiphertext &dest,
                   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
    void add_const_inplace(PhantomCiphertext &ct,
                           double value,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void add_reduced_error(const PhantomCiphertext &ct1,
                           const PhantomCiphertext &ct2,
                           PhantomCiphertext &dest,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
    void add_inplace_reduced_error(PhantomCiphertext &ct1,
                                   const PhantomCiphertext &ct2,
                                   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void sub_reduced_error(const PhantomCiphertext &ct1,
                           const PhantomCiphertext &ct2,
                           PhantomCiphertext &dest,
                           const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
    void sub_inplace_reduced_error(PhantomCiphertext &ct1,
                                   const PhantomCiphertext &ct2,
                                   const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void multiply_reduced_error(const PhantomCiphertext &ct1,
                                const PhantomCiphertext &ct2,
                                const PhantomRelinKey &relin_keys,
                                PhantomCiphertext &dest,
                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
    void multiply_inplace_reduced_error(PhantomCiphertext &ct1,
                                        const PhantomCiphertext &ct2,
                                        const PhantomRelinKey &relin_keys,
                                        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;
};

}
