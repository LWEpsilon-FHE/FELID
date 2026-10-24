#pragma once
using namespace std;
using namespace seal;

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

inline Ciphertext exp(const Ciphertext & x, const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Plaintext inverse_128;
  encoder.encode(0.0078125,x.parms_id(),x.scale(),inverse_128);

  Ciphertext output;
  evaluator.multiply_plain(x, inverse_128, output);
  evaluator.rescale_to_next_inplace(output);

  Plaintext one;
  encoder.encode(1.0, output.parms_id(), output.scale(), one);
  
  evaluator.add_plain_inplace(output, one);

  for (int i = 0; i < log2(128); ++i){
    
    evaluator.square_inplace(output);
    evaluator.relinearize_inplace(output, relin_keys);
    evaluator.rescale_to_next_inplace(output);
  }

  return output;

}

inline Ciphertext inverse(const Ciphertext & x, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, int iter) {
  
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Plaintext one;
  encoder.encode(1.0,x.parms_id(), x.scale(), one);

  Ciphertext y;
  evaluator.sub_plain(x, one, y);
  evaluator.negate_inplace(y);

  Ciphertext tmp;
  evaluator.add_plain(y, one, tmp);

  Ciphertext res = tmp;

  for (int i = 0; i < iter; ++i){
    evaluator.square_inplace(y);
    evaluator.relinearize_inplace(y,relin_keys);
    evaluator.rescale_to_next_inplace(y);

    encoder.encode(1.0,y.parms_id(), y.scale(), one);
    evaluator.add_plain(y, one, tmp);

    evaluator.mod_switch_to_inplace(res, tmp.parms_id());
    evaluator.multiply_inplace(res,tmp);
    evaluator.relinearize_inplace(res, relin_keys);
    evaluator.rescale_to_next_inplace(res);

  }

  return res;
}

inline void softmax_mod_switch_to_common_level(Ciphertext& a, Ciphertext& b,
    const SEALContext& context, Evaluator& evaluator) {
  if (context.get_context_data(a.parms_id())->chain_index() >
      context.get_context_data(b.parms_id())->chain_index())
    evaluator.mod_switch_to_inplace(a, b.parms_id());
  else if (a.parms_id() != b.parms_id())
    evaluator.mod_switch_to_inplace(b, a.parms_id());
}

inline Ciphertext softmax(const Ciphertext & enc_X, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, const GaloisKeys &RotK, int iter, double max){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  Ciphertext x_minus_max;
  Plaintext ecd_max;
  encoder.encode(max, enc_X.parms_id(),enc_X.scale(),ecd_max);
  evaluator.sub_plain(enc_X,ecd_max,x_minus_max);

  Ciphertext exp_x = exp(x_minus_max,seal_context,relin_keys);

  Ciphertext exp_sum = exp_x;
  evaluator.rotate_vector_inplace(exp_sum,n/2,RotK);
  evaluator.add_inplace(exp_sum,exp_x);

  Ciphertext inv_exp = inverse(exp_sum,seal_context,relin_keys,iter);

  softmax_mod_switch_to_common_level(exp_x, inv_exp, seal_context, evaluator);

  evaluator.multiply_inplace(exp_x,inv_exp);
  evaluator.relinearize_inplace(exp_x,relin_keys);
  evaluator.rescale_to_next_inplace(exp_x);

  return exp_x;

}

inline Ciphertext softmax_boot(const Ciphertext & enc_X, const SEALContext& seal_context, 
  const RelinKeys &relin_keys, const GaloisKeys &RotK, Bootstrapper & bootstrapper, int iter, double max){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  Ciphertext x_minus_max;
  Plaintext ecd_max;
  encoder.encode(max, enc_X.parms_id(),enc_X.scale(),ecd_max);
  evaluator.sub_plain(enc_X,ecd_max,x_minus_max);

  Ciphertext exp_x = exp(x_minus_max,seal_context,relin_keys);

  Ciphertext exp_sum = exp_x;
  evaluator.rotate_vector_inplace(exp_sum,n/2,RotK);
  evaluator.add_inplace(exp_sum,exp_x);

  Ciphertext exp_sum_boot = exp_sum;
  while (seal_context.get_context_data(exp_sum_boot.parms_id())->chain_index() > 0)
    evaluator.mod_switch_to_next_inplace(exp_sum_boot);
  Ciphertext rtn;
  bootstrapper.bootstrap_3(rtn,exp_sum_boot);
  exp_sum=rtn;

  Ciphertext inv_exp = inverse(exp_sum,seal_context,relin_keys,iter);

  softmax_mod_switch_to_common_level(exp_x, inv_exp, seal_context, evaluator);

  evaluator.multiply_inplace(exp_x,inv_exp);
  evaluator.relinearize_inplace(exp_x,relin_keys);
  evaluator.rescale_to_next_inplace(exp_x);   

  return exp_x;

}
