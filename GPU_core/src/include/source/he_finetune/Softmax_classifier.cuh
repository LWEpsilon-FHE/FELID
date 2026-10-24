#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>

#include "phantom.h"
#include "source/ckks_evaluator_parallel.cuh"
#include "source/bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;
using namespace fhe;

inline vector<double> softmax_pt(const vector<double> & x){
  vector<double> exp_x(x.size());
  double sum = 0;
  for (int i = 0; i < x.size(); ++i){
    exp_x[i] = exp(x[i]);
    
    sum += exp_x[i];
  }
  
  double inv_sum = 1.0/sum;
  for (int i = 0; i < exp_x.size(); ++i){
    exp_x[i] *= inv_sum;
  }
  return exp_x;
}

inline PhantomCiphertext softmax_classifier_exp(PhantomCiphertext & x, PhantomContext & seal_context, const PhantomRelinKey &relin_keys){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  PhantomPlaintext inverse_128;
  encoder.encode(0.0078125,x.params_id(),x.scale(),inverse_128);

  PhantomCiphertext output;
  evaluator.multiply_plain(x, inverse_128, output);
  evaluator.rescale_to_next_inplace(output);

  PhantomPlaintext one;
  encoder.encode(1.0, output.params_id(), output.scale(), one);
  
  evaluator.add_plain_inplace(output, one);

  for (int i = 0; i < log2(128); ++i){
    
    evaluator.square_inplace(output);
    evaluator.relinearize_inplace(output, relin_keys);
    evaluator.rescale_to_next_inplace(output);
  }

  return output;

}

inline PhantomCiphertext softmax_classifier_inverse(PhantomCiphertext & x, PhantomContext & seal_context, 
  const PhantomRelinKey &relin_keys, int iter) {
  
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  PhantomPlaintext one;
  encoder.encode(1.0,x.params_id(), x.scale(), one);

  PhantomCiphertext y;
  evaluator.sub_plain(x, one, y);
  evaluator.negate_inplace(y);

  PhantomCiphertext tmp;
  evaluator.add_plain(y, one, tmp);

  PhantomCiphertext res = tmp;

  for (int i = 0; i < iter; ++i){
    evaluator.square_inplace(y);
    evaluator.relinearize_inplace(y,relin_keys);
    evaluator.rescale_to_next_inplace(y);

    encoder.encode(1.0,y.params_id(), y.scale(), one);
    evaluator.add_plain(y, one, tmp);

    evaluator.mod_switch_to_inplace(res, tmp.params_id());
    evaluator.multiply_inplace(res,tmp);
    evaluator.relinearize_inplace(res, relin_keys);
    evaluator.rescale_to_next_inplace(res);

  }

  return res;
}

inline void he_finetune_mod_switch_to_common_lower_level(PhantomCiphertext &a,
  PhantomCiphertext &b, PhantomContext &seal_context, Evaluator &evaluator) {
  auto a_depth = seal_context.get_context_data(a.params_id()).chain_depth();
  auto b_depth = seal_context.get_context_data(b.params_id()).chain_depth();

  if (a_depth > b_depth) {
    evaluator.mod_switch_to_inplace(a, b.params_id());
  } else if (b_depth > a_depth) {
    evaluator.mod_switch_to_inplace(b, a.params_id());
  }
}

inline PhantomCiphertext softmax(PhantomCiphertext & enc_X, PhantomContext & seal_context, 
  const PhantomRelinKey &relin_keys, PhantomGaloisKey &RotK, int iter, double max){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  PhantomCiphertext x_minus_max;
  PhantomPlaintext ecd_max;
  encoder.encode(max, enc_X.params_id(),enc_X.scale(),ecd_max);
  evaluator.sub_plain(enc_X,ecd_max,x_minus_max);

  PhantomCiphertext exp_x = softmax_classifier_exp(x_minus_max,seal_context,relin_keys);

  PhantomCiphertext exp_sum = exp_x;
  evaluator.rotate_vector_inplace(exp_sum,n/2,RotK);
  evaluator.add_inplace(exp_sum,exp_x);

  PhantomCiphertext inv_exp = softmax_classifier_inverse(exp_sum,seal_context,relin_keys,iter);

  he_finetune_mod_switch_to_common_lower_level(exp_x, inv_exp, seal_context, evaluator);

  evaluator.multiply_inplace(exp_x,inv_exp);
  evaluator.relinearize_inplace(exp_x,relin_keys);
  evaluator.rescale_to_next_inplace(exp_x);

  return exp_x;

}

inline PhantomCiphertext softmax_boot(PhantomCiphertext & enc_X, PhantomContext & seal_context, 
  const PhantomRelinKey &relin_keys, PhantomGaloisKey &RotK, Bootstrapper & bootstrapper, int iter, double max){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  PhantomCiphertext x_minus_max;
  PhantomPlaintext ecd_max;
  encoder.encode(max, enc_X.params_id(),enc_X.scale(),ecd_max);
  evaluator.sub_plain(enc_X,ecd_max,x_minus_max);

  PhantomCiphertext exp_x = softmax_classifier_exp(x_minus_max,seal_context,relin_keys);

  PhantomCiphertext exp_sum = exp_x;
  evaluator.rotate_vector_inplace(exp_sum,n/2,RotK);
  evaluator.add_inplace(exp_sum,exp_x);

  PhantomCiphertext exp_sum_boot = exp_sum;
  while (seal_context.get_context_data(exp_sum_boot.params_id()).chain_depth() > 0) {
    evaluator.mod_switch_to_next_inplace(exp_sum_boot);
  }

  PhantomCiphertext rtn;
  bootstrapper.bootstrap_3(rtn,exp_sum_boot);
  exp_sum=rtn;

  PhantomCiphertext inv_exp = softmax_classifier_inverse(exp_sum,seal_context,relin_keys,iter);

  he_finetune_mod_switch_to_common_lower_level(exp_x, inv_exp, seal_context, evaluator);

  evaluator.multiply_inplace(exp_x,inv_exp);
  evaluator.relinearize_inplace(exp_x,relin_keys);
  evaluator.rescale_to_next_inplace(exp_x);   

  return exp_x;

}
