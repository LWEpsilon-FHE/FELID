#pragma once
#include "Layernorm_affine.hpp"
#include "Bootstrapper.h"

using namespace std;
using namespace seal;

inline Ciphertext invert_sqrt_boot_v2(Ciphertext x, int d_newt, int d_gold, 
  const SEALContext& seal_context, const RelinKeys &relin_keys, Bootstrapper& bootstrapper){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Ciphertext res = initGuess(x, seal_context);    

  for (int j = 0 ; j < 3 ; ++j){
    evaluator.mod_switch_to_next_inplace(res);   
  }

  Ciphertext y = newtonIter(x,res,d_newt,seal_context,relin_keys);    

  while(seal_context.get_context_data(y.parms_id())->chain_index()>0){
  evaluator.mod_switch_to_next_inplace(y);
  }
  Ciphertext rtn;
  bootstrapper.bootstrap_3(rtn,y);   
  y=rtn;

  Ciphertext sqrt_inv = goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline vector<Ciphertext> layernorm_v2(const vector<Ciphertext> & x1, const vector<Ciphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta,double c, 
  int row, int col, int batch_size,
  const SEALContext& seal_context, const RelinKeys &relin_keys, const GaloisKeys & RotK){
  
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  validate_layernorm_shape(x1.size(), x2.size(), encoder.slot_count(),
      row, col, batch_size, gamma, beta, c);

  double scale = x1[0].scale();
  size_t slot_count = encoder.slot_count();
  int n = slot_count;
  int num_ct = x1.size();

  if(seal_context.get_context_data(x1[0].parms_id()) -> chain_index() != 
    seal_context.get_context_data(x2[0].parms_id()) -> chain_index()){
    
  }
  if(x1.size() != x2.size()){
    
  }

  vector<Ciphertext> x1x2(num_ct);
  for (int i = 0; i < num_ct; ++i){
    evaluator.add(x1[i],x2[i],x1x2[i]);
  }

  Ciphertext sum = x1x2[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(sum,x1x2[i]);
  }

  int sum_index = n/2;
  int min_index = row * batch_size;
  while(sum_index >= min_index){
    Ciphertext copy = sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(sum,copy);
    sum_index /= 2;
  }

  vector<double> vec_col(n,(double)col);
  Plaintext ecd_col;
  encoder.encode(vec_col, x1x2[0].parms_id(), x1x2[0].scale(), ecd_col);

  Ciphertext tmp_sum = sum;
  evaluator.mod_switch_to_next_inplace(tmp_sum);

  Ciphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    Ciphertext tmp;
    evaluator.multiply_plain(x1x2[i],ecd_col,tmp);
    evaluator.rescale_to_next_inplace(tmp);

    tmp.scale()=tmp_sum.scale();
    
    evaluator.sub_inplace(tmp,tmp_sum);
  
    evaluator.square_inplace(tmp);

    if(i == 0){
      var_sum = tmp;
    }
    else{
      evaluator.add_inplace(var_sum,tmp);
    }
  }

  evaluator.relinearize_inplace(var_sum,relin_keys);
  evaluator.rescale_to_next_inplace(var_sum);

  sum_index = n/2;
  min_index = row * batch_size;
  while(sum_index >= min_index){
    Ciphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*(double)(col)*(double)(col)*c));
  Plaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.parms_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum);

  Ciphertext inv_sqrt_var = invert_sqrt(var_sum,4,2,seal_context,relin_keys);

  for (int i = 0; i < num_ct; ++i){
    
    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    Plaintext ecd_gamma;
    
    encoder.encode(gamma_c,x1x2[i].parms_id(),x1x2[i].scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(x1x2[i],ecd_gamma);
    evaluator.rescale_to_next_inplace(x1x2[i]);

    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/((double)col*sqrt(c));
    }
    Ciphertext tmp_sum = sum;
    encoder.encode(gamma_c,tmp_sum.parms_id(),tmp_sum.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(tmp_sum,ecd_gamma);
    evaluator.rescale_to_next_inplace(tmp_sum);

    evaluator.sub_inplace(x1x2[i],tmp_sum);

    evaluator.mod_switch_to_inplace(x1x2[i], inv_sqrt_var.parms_id());
    x1x2[i].scale()=inv_sqrt_var.scale();
    evaluator.multiply_inplace(x1x2[i],inv_sqrt_var);
    evaluator.relinearize_inplace(x1x2[i],relin_keys);
    evaluator.rescale_to_next_inplace(x1x2[i]);

    vector<double> beta_i(n);
    for (int j = 0; j < n; ++j)
      beta_i[j] = beta[layernorm_feature(i, j, n, row, batch_size)];
    Plaintext ecd_beta;
    encoder.encode(beta_i,x1x2[i].parms_id(),x1x2[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(x1x2[i],ecd_beta);
  }

  return x1x2;

}

inline vector<Ciphertext> layernorm_boot_self_v2(const vector<Ciphertext> & x1, const vector<Ciphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta, double c,
   int row, int col, int batch_size,
  const SEALContext& seal_context, const RelinKeys &relin_keys, const GaloisKeys & RotK, Bootstrapper& bootstrapper){
  
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  validate_layernorm_shape(x1.size(), x2.size(), encoder.slot_count(),
      row, col, batch_size, gamma, beta, c);

  double scale = x1[0].scale();
  size_t slot_count = encoder.slot_count();
  int n = slot_count;
  int num_ct = x1.size();

  if(seal_context.get_context_data(x1[0].parms_id()) -> chain_index() != 
    seal_context.get_context_data(x2[0].parms_id()) -> chain_index()){
    
  }
  if(x1.size() != x2.size()){
    
  }

  vector<Ciphertext> x1x2(num_ct);
  for (int i = 0; i < num_ct; ++i){
    evaluator.add(x1[i],x2[i],x1x2[i]);
  }

  Ciphertext sum = x1x2[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(sum,x1x2[i]);
  }

  int sum_index = n/2;
  int min_index = row * batch_size;
  while(sum_index >= min_index){
    Ciphertext copy = sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(sum,copy);
    sum_index /= 2;
  }

  vector<double> vec_col(n,(double)col);
  Plaintext ecd_col;
  encoder.encode(vec_col, x1x2[0].parms_id(), x1x2[0].scale(), ecd_col);

  Ciphertext tmp_sum = sum;
  evaluator.mod_switch_to_next_inplace(tmp_sum);

  Ciphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    Ciphertext tmp;
    evaluator.multiply_plain(x1x2[i],ecd_col,tmp);   
    evaluator.rescale_to_next_inplace(tmp);

    tmp.scale()=tmp_sum.scale();
    
    evaluator.sub_inplace(tmp,tmp_sum);
  
    evaluator.square_inplace(tmp);   

    if(i == 0){
      var_sum = tmp;
    }
    else{
      evaluator.add_inplace(var_sum,tmp);
    }
  }

  evaluator.relinearize_inplace(var_sum,relin_keys);
  evaluator.rescale_to_next_inplace(var_sum);

  for (int i = 0; i < 4; ++i){
    evaluator.mod_switch_to_next_inplace(var_sum); 
  }

  sum_index = n/2;
  min_index = row * batch_size;
  while(sum_index >= min_index){
    Ciphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*(double)(col)*(double)(col)*c));
  Plaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.parms_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum); 

  Ciphertext rtn;
  bootstrapper.bootstrap_3(rtn,var_sum);   
  var_sum=rtn;

  Ciphertext inv_sqrt_var = invert_sqrt_boot(var_sum,4,2,seal_context,relin_keys,bootstrapper);  

  for (int i = 0; i < num_ct; ++i){
    
    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    Plaintext ecd_gamma;
    
    encoder.encode(gamma_c,x1x2[i].parms_id(),x1x2[i].scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(x1x2[i],ecd_gamma);
    evaluator.rescale_to_next_inplace(x1x2[i]);  
 
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/((double)col*sqrt(c));
    }
    Ciphertext tmp_sum = sum;
    encoder.encode(gamma_c,tmp_sum.parms_id(),tmp_sum.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(tmp_sum,ecd_gamma);
    evaluator.rescale_to_next_inplace(tmp_sum);   

    evaluator.sub_inplace(x1x2[i],tmp_sum);

    x1x2[i].scale()=inv_sqrt_var.scale();
    evaluator.multiply_inplace(x1x2[i],inv_sqrt_var);  
    evaluator.relinearize_inplace(x1x2[i],relin_keys);
    evaluator.rescale_to_next_inplace(x1x2[i]);   

    vector<double> beta_i(n);
    for (int j = 0; j < n; ++j)
      beta_i[j] = beta[layernorm_feature(i, j, n, row, batch_size)];
    Plaintext ecd_beta;
    encoder.encode(beta_i,x1x2[i].parms_id(),x1x2[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(x1x2[i],ecd_beta);
  }

  return x1x2;

}
