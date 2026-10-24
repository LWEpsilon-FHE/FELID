#include "RbootSparseOps.cuh"

#include <stdexcept>

using namespace phantom;
using namespace phantom::util;
using namespace moai;

namespace rboot_slim {

RbootSparseOps::RbootSparseOps(CKKSEvaluator *ckks,
                               PhantomCKKSEncoder *raw_encoder,
                               size_t configured_sparse_slots)
    : ckks(ckks),
      raw_encoder(raw_encoder),
      configured_sparse_slots(configured_sparse_slots)
{
    if (this->configured_sparse_slots == 0 && ckks != nullptr)
    {
        this->configured_sparse_slots = ckks->context->first_context_data().parms().sparse_slots();
    }
}

size_t RbootSparseOps::full_slot_count() const
{
    return raw_encoder->slot_count();
}

size_t RbootSparseOps::sparse_slot_count() const
{
    if (configured_sparse_slots != 0)
    {
        return configured_sparse_slots;
    }
    return raw_encoder->message_length();
}

void RbootSparseOps::multiply_const(const PhantomCiphertext &ct,
                                    double value,
                                    PhantomCiphertext &dest,
                                    const cuda_stream_wrapper &stream_wrapper) const
{
    dest = ct;
    multiply_const_inplace(dest, value, stream_wrapper);
}

void RbootSparseOps::multiply_const_inplace(PhantomCiphertext &ct,
                                            double value,
                                            const cuda_stream_wrapper &stream_wrapper) const
{
    PhantomPlaintext const_plain;
    std::vector<double> values(full_slot_count(), value);
    encode_exact_vector(values, ct.scale(), const_plain, 1, stream_wrapper);
    ckks->evaluator.mod_switch_to_inplace(const_plain, ct.params_id(), stream_wrapper);
    ckks->evaluator.multiply_plain_inplace(ct, const_plain, stream_wrapper);
}

void RbootSparseOps::add_const(const PhantomCiphertext &ct,
                               double value,
                               PhantomCiphertext &dest,
                               const cuda_stream_wrapper &stream_wrapper) const
{
    dest = ct;
    add_const_inplace(dest, value, stream_wrapper);
}

void RbootSparseOps::add_const_inplace(PhantomCiphertext &ct,
                                       double value,
                                       const cuda_stream_wrapper &stream_wrapper) const
{
    PhantomPlaintext const_plain;
    std::vector<double> values(full_slot_count(), value);
    encode_exact_vector(values, ct.scale(), const_plain, 1, stream_wrapper);
    ckks->evaluator.mod_switch_to_inplace(const_plain, ct.params_id(), stream_wrapper);
    ckks->evaluator.add_plain_inplace(ct, const_plain, stream_wrapper);
}

void RbootSparseOps::add_reduced_error(const PhantomCiphertext &ct1,
                                       const PhantomCiphertext &ct2,
                                       PhantomCiphertext &dest,
                                       const cuda_stream_wrapper &stream_wrapper) const
{
    if (&ct2 == &dest)
    {
        add_inplace_reduced_error(dest, ct1, stream_wrapper);
    }
    else
    {
        dest = ct1;
        add_inplace_reduced_error(dest, ct2, stream_wrapper);
    }
}

void RbootSparseOps::add_inplace_reduced_error(PhantomCiphertext &ct1,
                                               const PhantomCiphertext &ct2,
                                               const cuda_stream_wrapper &stream_wrapper) const
{
    const size_t ct1_mod_size = ct1.coeff_modulus_size();
    const size_t ct2_mod_size = ct2.coeff_modulus_size();

    if (ct1_mod_size == ct2_mod_size)
    {
        ct1.scale() = ct2.scale();
        ckks->evaluator.add_inplace(ct1, ct2, stream_wrapper);
        return;
    }

    if (ct1_mod_size < ct2_mod_size)
    {
        const auto &context_data = ckks->context->get_context_data(ct2.params_id());
        const auto modulus = context_data.parms().coeff_modulus();
        PhantomCiphertext adjusted;

        const double scale_adjust = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value()) /
                                    (ct2.scale() * ct2.scale());
        multiply_const(ct2, scale_adjust, adjusted, stream_wrapper);
        adjusted.scale() = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value());
        ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
        ckks->evaluator.mod_switch_to_inplace(adjusted, ct1.params_id(), stream_wrapper);
        ct1.scale() = adjusted.scale();
        ckks->evaluator.add_inplace(ct1, adjusted, stream_wrapper);
        return;
    }

    const auto &context_data = ckks->context->get_context_data(ct1.params_id());
    const auto modulus = context_data.parms().coeff_modulus();
    PhantomCiphertext adjusted;

    const double scale_adjust = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value()) /
                                (ct1.scale() * ct1.scale());
    multiply_const(ct1, scale_adjust, adjusted, stream_wrapper);
    adjusted.scale() = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value());
    ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
    ckks->evaluator.mod_switch_to_inplace(adjusted, ct2.params_id(), stream_wrapper);
    adjusted.scale() = ct2.scale();
    ckks->evaluator.add(adjusted, ct2, ct1, stream_wrapper);
}

void RbootSparseOps::sub_reduced_error(const PhantomCiphertext &ct1,
                                       const PhantomCiphertext &ct2,
                                       PhantomCiphertext &dest,
                                       const cuda_stream_wrapper &stream_wrapper) const
{
    dest = ct1;
    sub_inplace_reduced_error(dest, ct2, stream_wrapper);
}

void RbootSparseOps::sub_inplace_reduced_error(PhantomCiphertext &ct1,
                                               const PhantomCiphertext &ct2,
                                               const cuda_stream_wrapper &stream_wrapper) const
{
    const size_t ct1_mod_size = ct1.coeff_modulus_size();
    const size_t ct2_mod_size = ct2.coeff_modulus_size();

    if (ct1_mod_size == ct2_mod_size)
    {
        ct1.scale() = ct2.scale();
        ckks->evaluator.sub_inplace(ct1, ct2, stream_wrapper);
        return;
    }

    if (ct1_mod_size < ct2_mod_size)
    {
        const auto &context_data = ckks->context->get_context_data(ct2.params_id());
        const auto modulus = context_data.parms().coeff_modulus();
        PhantomCiphertext adjusted;

        const double scale_adjust = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value()) /
                                    (ct2.scale() * ct2.scale());
        multiply_const(ct2, scale_adjust, adjusted, stream_wrapper);
        adjusted.scale() = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value());
        ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
        ckks->evaluator.mod_switch_to_inplace(adjusted, ct1.params_id(), stream_wrapper);
        ct1.scale() = adjusted.scale();
        ckks->evaluator.sub_inplace(ct1, adjusted, stream_wrapper);
        return;
    }

    const auto &context_data = ckks->context->get_context_data(ct1.params_id());
    const auto modulus = context_data.parms().coeff_modulus();
    PhantomCiphertext adjusted;

    const double scale_adjust = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value()) /
                                (ct1.scale() * ct1.scale());
    multiply_const(ct1, scale_adjust, adjusted, stream_wrapper);
    adjusted.scale() = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value());
    ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
    ckks->evaluator.mod_switch_to_inplace(adjusted, ct2.params_id(), stream_wrapper);
    adjusted.scale() = ct2.scale();
    ckks->evaluator.sub(adjusted, ct2, ct1, stream_wrapper);
}

void RbootSparseOps::multiply_reduced_error(const PhantomCiphertext &ct1,
                                            const PhantomCiphertext &ct2,
                                            const PhantomRelinKey &relin_keys,
                                            PhantomCiphertext &dest,
                                            const cuda_stream_wrapper &stream_wrapper) const
{
    if (&ct2 == &dest)
    {
        multiply_inplace_reduced_error(dest, ct1, relin_keys, stream_wrapper);
    }
    else
    {
        dest = ct1;
        multiply_inplace_reduced_error(dest, ct2, relin_keys, stream_wrapper);
    }
}

void RbootSparseOps::multiply_inplace_reduced_error(PhantomCiphertext &ct1,
                                                    const PhantomCiphertext &ct2,
                                                    const PhantomRelinKey &relin_keys,
                                                    const cuda_stream_wrapper &stream_wrapper) const
{
    const size_t ct1_mod_size = ct1.coeff_modulus_size();
    const size_t ct2_mod_size = ct2.coeff_modulus_size();

    if (ct1_mod_size == ct2_mod_size)
    {
        ct1.scale() = ct2.scale();
        ckks->evaluator.multiply_inplace(ct1, ct2, stream_wrapper);
        ckks->evaluator.relinearize_inplace(ct1, relin_keys, stream_wrapper);
        return;
    }

    if (ct1_mod_size < ct2_mod_size)
    {
        const auto &context_data = ckks->context->get_context_data(ct2.params_id());
        const auto modulus = context_data.parms().coeff_modulus();
        PhantomCiphertext adjusted;

        const double scale_adjust = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value()) /
                                    (ct2.scale() * ct2.scale());
        multiply_const(ct2, scale_adjust, adjusted, stream_wrapper);
        adjusted.scale() = ct1.scale() * static_cast<double>(modulus[ct2_mod_size - 1].value());
        ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
        ckks->evaluator.mod_switch_to_inplace(adjusted, ct1.params_id(), stream_wrapper);
        ct1.scale() = adjusted.scale();
        ckks->evaluator.multiply_inplace(ct1, adjusted, stream_wrapper);
    }
    else
    {
        const auto &context_data = ckks->context->get_context_data(ct1.params_id());
        const auto modulus = context_data.parms().coeff_modulus();
        PhantomCiphertext adjusted;

        const double scale_adjust = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value()) /
                                    (ct1.scale() * ct1.scale());
        multiply_const(ct1, scale_adjust, adjusted, stream_wrapper);
        adjusted.scale() = ct2.scale() * static_cast<double>(modulus[ct1_mod_size - 1].value());
        ckks->evaluator.rescale_to_next_inplace(adjusted, stream_wrapper);
        ckks->evaluator.mod_switch_to_inplace(adjusted, ct2.params_id(), stream_wrapper);
        adjusted.scale() = ct2.scale();
        ckks->evaluator.multiply(adjusted, ct2, ct1, stream_wrapper);
    }

    ckks->evaluator.relinearize_inplace(ct1, relin_keys, stream_wrapper);
}

}
