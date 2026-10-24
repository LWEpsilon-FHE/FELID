#pragma once
#include "include.cuh"

__global__ void Kernel_modraise_inplace(uint64_t *poly_dest, const uint64_t *poly_src_zero, const uint64_t modulus,
                                        const uint64_t minus_q0_modulus, size_t N, uint64_t q0)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    auto q = modulus;
    poly_dest[i] = poly_src_zero[i] % q;
    if (poly_src_zero[i] > (q0 >> 1))
    {
      poly_dest[i] += minus_q0_modulus;
      poly_dest[i] -= (poly_dest[i] >= q) ? q : 0;
    }
  }
}



void ModRaise_inplace(PhantomCiphertext & cipher, PhantomContext& context){

  PhantomCKKSEncoder phantom_encoder(context);
  Encoder encoder(&context, &phantom_encoder);
  Evaluator evaluator(&context, &phantom_encoder);
  //CKKSEncoder encoder(seal_context);
  //  cout <<"scale of ct: "<<log2(cipher.scale())<<endl;

if (cipher.size() != 2)
  {
    throw invalid_argument("Ciphertexts of size 2 are supported only!");
  }

  if (cipher.coeff_modulus_size() != 1)
  {
    throw invalid_argument("Ciphertexts in the lowest level are supported only!");
  }

  if (cipher.is_ntt_form())
  {
    evaluator.transform_from_ntt_inplace(cipher);
  }

  const auto &stream = phantom::util::global_variables::default_stream->get_stream();
  auto N = cipher.poly_modulus_degree();
  const auto &context_data = context.first_context_data();
  const auto &modulus = context_data.parms().coeff_modulus();
  auto mod_count = modulus.size();
  auto rns_coeff_count = mod_count * N;

  uint64_t q0 = modulus[0].value();
  vector<uint64_t> minus_q0(mod_count);
  minus_q0[0] = 0;

  for (size_t l = 1; l < mod_count; l++)
  {
    minus_q0[l] = modulus[l].value() - q0 % modulus[l].value();
  }

  // Resize to the full level.
  cipher.resize(context, context_data.chain_index(), stream);

  auto ciphertext_size = cipher.size();
  PhantomCiphertext cipher_copy = cipher;

  uint64_t gridDimGlb = N / blockDimGlb.x;
  for (size_t poly_idx = 0; poly_idx < ciphertext_size; poly_idx++)
  {
    const auto rns_poly_src = cipher_copy.data() + poly_idx * rns_coeff_count;
    const auto rns_poly_dest = cipher.data() + poly_idx * rns_coeff_count;

    for (size_t j = 0; j < mod_count; j++)
    {
      const auto poly_dest = rns_poly_dest + j * N;
      Kernel_modraise_inplace<<<gridDimGlb, blockDimGlb, 0, stream>>>(
          poly_dest, rns_poly_src, modulus[j].value(), minus_q0[j], N, q0);
    }
  }

    evaluator.transform_to_ntt_inplace(cipher);
}



// using namespace std;
// using namespace seal;
// using namespace seal::util;

// void ModRaise_inplace(Ciphertext & cipher, const SEALContext& context){

//   CKKSEncoder encoder(context);
//   Evaluator evaluator(context,encoder);
//   //CKKSEncoder encoder(seal_context);
//   //  cout <<"scale of ct: "<<log2(cipher.scale())<<endl;

//     if (cipher.size() != 2) {
//     throw invalid_argument("Ciphertexts of size 2 are supported only!");
//   }

//   if (cipher.coeff_modulus_size() != 1) {
//     throw invalid_argument("Ciphertexts in the lowest level are supported only!");
//   }

//   if (cipher.is_ntt_form()) {
//     evaluator.transform_from_ntt_inplace(cipher);
//   }

//   auto parms_id = context.first_parms_id();

//   // Make a copy of ciphertext
//   Ciphertext encrypted_copy(cipher);

//   // Resize to the full level.
//   cipher.resize(context, parms_id, 2);

//   auto ciphertext_size = cipher.size();
//   const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
//   auto coeff_modulus_size = cipher.coeff_modulus_size();
//   auto poly_modulus_degree = cipher.poly_modulus_degree();

//   uint64_t q0 = modulus[0].value();
//   cout <<"q0 in ModRaise"<<(double)q0<<", log = "<<log2((double)q0)<<endl;

//   vector<uint64_t> minus_q0(coeff_modulus_size);
//   minus_q0[0] = 0;

//   for (size_t l = 1; l < coeff_modulus_size; l++) {
//     minus_q0[l] = modulus[l].value() - q0 % modulus[l].value();
//   }

//   for (size_t poly_idx = 0; poly_idx < ciphertext_size; poly_idx++) {
//     const auto &rns_poly_src = iter(encrypted_copy)[poly_idx];
//     const auto &rns_poly_dest = iter(cipher)[poly_idx];
//     const auto &poly_src_zero = rns_poly_src[0];

//     for (size_t j = 0; j < coeff_modulus_size; j++) {
//       const auto &poly_dest = rns_poly_dest[j];
//       for (size_t i = 0; i < poly_modulus_degree; i++) {
//         auto q = modulus[j].value();
//         poly_dest[i] = poly_src_zero[i] % q;
//         if (poly_src_zero[i] > (q0 >> 1)) {
//           poly_dest[i] += minus_q0[j];
//           poly_dest[i] -= (poly_dest[i] >= q) ? q : 0;
//         }
//       }
//     }
//   }

//   // Switch back to the NTT form.
//   evaluator.transform_to_ntt_inplace(cipher);

//   //const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
//  // cout <<"scale of ct: "<<log2(cipher.scale())<<endl;

//   //cipher.scale() = ((double)modulus[0].value());

//   //cout <<"scale of ct: "<<log2(cipher.scale())<<endl;
// }
