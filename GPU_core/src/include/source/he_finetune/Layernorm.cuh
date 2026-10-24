#pragma once
#include "Layernorm_affine.cuh"
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

inline vector<vector<double>> layernorm_pt(const vector<vector<double>> & input1, const vector<vector<double>> & input2, 
  int row, int col, 
  const vector<double>& gamma, const vector<double>& beta){
  vector<double> ave(row,0);
  for (int i = 0; i < row; ++i){
    for (int j = 0 ; j < col ; ++j){
      ave[i] += (input1[i][j]+input2[i][j])/(double)col;
    }
  }

  vector<double> var(row,0);
  for (int i = 0; i < row; ++i){
    for (int j = 0 ; j < col ; ++j){
      var[i] += (input1[i][j]+input2[i][j]-ave[i])*(input1[i][j]+input2[i][j]-ave[i]);
    }
    var[i] /= (double)col;
    var[i] = 1.0/sqrt(var[i]);
  }

  vector<vector<double>> output(row, vector<double>(col,0));
  for (int i = 0; i < row; ++i){
    for (int j = 0 ; j < col ; ++j){
      output[i][j] = gamma[j]*var[i]*(input1[i][j]+input2[i][j]-ave[i])+beta[j];
    }
  }

  return output;
}

inline PhantomCiphertext he_finetune_evalLine(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c, PhantomContext & seal_context){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  double scale = x.scale();

  evaluator.mod_switch_to_inplace(m,x.params_id());
  evaluator.multiply_plain_inplace(x,m);
  evaluator.rescale_to_next_inplace(x);
  evaluator.mod_switch_to_inplace(c,x.params_id());
  x.scale() = scale;
  evaluator.add_plain_inplace(x,c);
  return x;
}

inline PhantomCiphertext he_finetune_initGuess(PhantomCiphertext x, PhantomContext & seal_context){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  PhantomPlaintext a,b;
  encoder.encode(-1.29054537e-04, x.scale(), a);
  encoder.encode(1.29054537e-01, x.scale(), b);
  return he_finetune_evalLine(x,a,b,seal_context);
}

inline PhantomCiphertext he_finetune_newtonIter(PhantomCiphertext x, PhantomCiphertext res, int iter, 
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  double scale = x.scale();

  for (int i = 0; i < iter; ++i){
    PhantomPlaintext three_half, neg_half;
    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    PhantomCiphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq,relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    PhantomCiphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half,x.params_id());
    evaluator.multiply_plain(x,neg_half,res_x);
    evaluator.rescale_to_next_inplace(res_x);
    if(seal_context.get_context_data(res.params_id()).chain_depth()<
      seal_context.get_context_data(res_x.params_id()).chain_depth()){
      evaluator.mod_switch_to_inplace(res_x,res.params_id());
    }
    else{
      evaluator.mod_switch_to_inplace(res,res_x.params_id());
    }

    evaluator.multiply_inplace(res_x,res);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(res_sq,res_x.params_id());
    evaluator.multiply_inplace(res_x,res_sq);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(three_half, res.params_id());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    evaluator.mod_switch_to_inplace(res, res_x.params_id());
    res_x.scale()=scale;
    res.scale()=scale;
    evaluator.add_inplace(res, res_x);
  }
  return res;
}

inline PhantomCiphertext he_finetune_goldSchmidtIter(PhantomCiphertext v, PhantomCiphertext y, int d, 
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  double scale = y.scale();

  PhantomPlaintext constant;
  encoder.encode(0.5,scale,constant);

  evaluator.mod_switch_to_inplace(v,y.params_id());
  PhantomCiphertext x;
  evaluator.multiply(v,y,x);
  evaluator.relinearize_inplace(x,relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant,y.params_id());
  PhantomCiphertext h;
  evaluator.multiply_plain(y,constant,h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; ++i){
    encoder.encode(0.5,scale,constant);
    PhantomCiphertext r;
    evaluator.multiply(x,h,r);
    evaluator.relinearize_inplace(r,relin_keys);
    evaluator.rescale_to_next_inplace(r);
    r.scale() = scale;

    PhantomCiphertext temp;
    evaluator.negate(r,temp);
    evaluator.mod_switch_to_inplace(constant,temp.params_id());
    evaluator.add_plain(temp, constant, r);

    evaluator.mod_switch_to_inplace(x,r.params_id());
    evaluator.multiply(x,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    x.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(x,temp.params_id());
    evaluator.add_inplace(x,temp);

    evaluator.mod_switch_to_inplace(h,r.params_id());
    evaluator.multiply(h,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    h.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(h,temp.params_id());
    evaluator.add_inplace(h,temp);
  }
  encoder.encode(2.0,scale,constant);
  evaluator.mod_switch_to_inplace(constant, h.params_id());
  evaluator.multiply_plain_inplace(h,constant);
  evaluator.rescale_to_next_inplace(h);

  return h;

}

inline PhantomCiphertext he_finetune_invert_sqrt(PhantomCiphertext x, int d_newt, int d_gold, 
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  PhantomCiphertext res = he_finetune_initGuess(x, seal_context);    

  PhantomCiphertext y = he_finetune_newtonIter(x,res,d_newt,seal_context,relin_keys);    

  PhantomCiphertext sqrt_inv = he_finetune_goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline PhantomCiphertext invert_sqrt_boot(PhantomCiphertext x, int d_newt, int d_gold, 
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys, Bootstrapper& bootstrapper){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  PhantomCiphertext res = he_finetune_initGuess(x, seal_context);    

  for (int j = 0 ; j < 3 ; ++j){
    evaluator.mod_switch_to_next_inplace(res);   
  }

  PhantomCiphertext y = he_finetune_newtonIter(x,res,d_newt,seal_context,relin_keys);    

  while(seal_context.get_context_data(y.params_id()).chain_depth()>0){
  evaluator.mod_switch_to_next_inplace(y);
  }
  PhantomCiphertext rtn;
  bootstrapper.bootstrap_3(rtn,y);   
  y=rtn;

  PhantomCiphertext sqrt_inv = he_finetune_goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline PhantomCiphertext invert_sqrt_boot2(PhantomCiphertext x, int d_newt, int d_gold, 
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys, Bootstrapper& bootstrapper){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  PhantomCiphertext res = he_finetune_initGuess(x, seal_context);    

  for (int j = 0 ; j < 1 ; ++j){
    evaluator.mod_switch_to_next_inplace(res);   
  }

  PhantomCiphertext y = he_finetune_newtonIter(x,res,d_newt,seal_context,relin_keys);    

  while(seal_context.get_context_data(y.params_id()).chain_depth()>0){
  evaluator.mod_switch_to_next_inplace(y);
  }
  PhantomCiphertext rtn;
  bootstrapper.bootstrap_3(rtn,y);   
  y=rtn;

  for (int j = 0 ; j < 4 ; ++j){
    evaluator.mod_switch_to_next_inplace(y);   
  }

  PhantomCiphertext sqrt_inv = he_finetune_goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline vector<PhantomCiphertext> layernorm(vector<PhantomCiphertext> & x1, vector<PhantomCiphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta, double c,
   int row, int col, int batch_size,
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys, PhantomGaloisKey & RotK){
  
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  validate_layernorm_shape(x1.size(), x2.size(), encoder.slot_count(),
      row, col, batch_size, gamma, beta, c);

  double scale = x1[0].scale();
  size_t slot_count = encoder.slot_count();
  int n = slot_count;
  int num_ct = x1.size();

  if(seal_context.get_context_data(x1[0].params_id()).chain_depth() != 
    seal_context.get_context_data(x2[0].params_id()).chain_depth()){
    
  }
  if(x1.size() != x2.size()){
    
  }

  vector<PhantomCiphertext> x1x2(num_ct);
  for (int i = 0; i < num_ct; ++i){
    evaluator.add(x1[i],x2[i],x1x2[i]);
  }

  PhantomCiphertext sum = x1x2[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(sum,x1x2[i]);
  }

  int sum_index = n/2;
  int min_index = row * batch_size;
  while(sum_index >= min_index){
    PhantomCiphertext copy = sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_col(n, 1.0/(double)col);
  PhantomPlaintext ecd_inv_col;
  encoder.encode(inv_col, sum.params_id(), sum.scale(), ecd_inv_col);
  PhantomCiphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  PhantomCiphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);
    PhantomCiphertext tmp;
    evaluator.square(x1x2[i],tmp);

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
    PhantomCiphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
  PhantomPlaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.params_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum);

  PhantomCiphertext inv_sqrt_var = he_finetune_invert_sqrt(var_sum,4,2,seal_context,relin_keys);

  for (int i = 0; i < num_ct; ++i){

    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    PhantomPlaintext ecd_gamma;
    
    encoder.encode(gamma_c,x1x2[i].params_id(),x1x2[i].scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(x1x2[i],ecd_gamma);
    evaluator.rescale_to_next_inplace(x1x2[i]);

    evaluator.mod_switch_to_inplace(x1x2[i], inv_sqrt_var.params_id());
    evaluator.multiply_inplace(x1x2[i],inv_sqrt_var);
    evaluator.relinearize_inplace(x1x2[i],relin_keys);
    evaluator.rescale_to_next_inplace(x1x2[i]);

    vector<double> beta_i(n);
    for (int j = 0; j < n; ++j)
      beta_i[j] = beta[layernorm_feature(i, j, n, row, batch_size)];
    PhantomPlaintext ecd_beta;
    encoder.encode(beta_i,x1x2[i].params_id(),x1x2[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(x1x2[i],ecd_beta);
  }

  return x1x2;

}

inline vector<PhantomCiphertext> layernorm_boot_self(vector<PhantomCiphertext> & x1, vector<PhantomCiphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta, double c,
   int row, int col, int batch_size,
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys, PhantomGaloisKey & RotK, Bootstrapper& bootstrapper){
  
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  validate_layernorm_shape(x1.size(), x2.size(), encoder.slot_count(),
      row, col, batch_size, gamma, beta, c);

  double scale = x1[0].scale();
  size_t slot_count = encoder.slot_count();
  int n = slot_count;
  int num_ct = x1.size();

  if(seal_context.get_context_data(x1[0].params_id()).chain_depth() != 
    seal_context.get_context_data(x2[0].params_id()).chain_depth()){
    
  }
  if(x1.size() != x2.size()){
    
  }

  vector<PhantomCiphertext> x1x2(num_ct);
  for (int i = 0; i < num_ct; ++i){
    evaluator.add(x1[i],x2[i],x1x2[i]);
  }

  PhantomCiphertext sum = x1x2[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(sum,x1x2[i]);
  }

  int sum_index = n/2;
  int min_index = row * batch_size;
  while(sum_index >= min_index){
    PhantomCiphertext copy = sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_col(n, 1.0/(double)col);
  PhantomPlaintext ecd_inv_col;
  encoder.encode(inv_col, sum.params_id(), sum.scale(), ecd_inv_col);
  PhantomCiphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  PhantomCiphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);           
    PhantomCiphertext tmp;
    evaluator.square(x1x2[i],tmp);

    if(i == 0){
      var_sum = tmp;
    }
    else{
      evaluator.add_inplace(var_sum,tmp);
    }
  }

  evaluator.relinearize_inplace(var_sum,relin_keys);
  evaluator.rescale_to_next_inplace(var_sum);    

  for (int i = 0; i < 3; ++i){
    evaluator.mod_switch_to_next_inplace(var_sum); 
  }

  sum_index = n/2;
  min_index = row * batch_size;
  while(sum_index >= min_index){
    PhantomCiphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
  PhantomPlaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.params_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum);   

  PhantomCiphertext rtn;
  bootstrapper.bootstrap_3(rtn,var_sum);   
  var_sum=rtn;

  PhantomCiphertext inv_sqrt_var = invert_sqrt_boot(var_sum,4,2,seal_context,relin_keys,bootstrapper);  

  for (int i = 0; i < num_ct; ++i){

    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    PhantomPlaintext ecd_gamma;

    PhantomCiphertext scaled_inverse = inv_sqrt_var;
    encoder.encode(gamma_c,scaled_inverse.params_id(),scaled_inverse.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(scaled_inverse,ecd_gamma);
    evaluator.rescale_to_next_inplace(scaled_inverse);

    evaluator.mod_switch_to_inplace(x1x2[i], scaled_inverse.params_id());
    evaluator.multiply_inplace(x1x2[i],scaled_inverse);
    evaluator.relinearize_inplace(x1x2[i],relin_keys);
    evaluator.rescale_to_next_inplace(x1x2[i]);  

    vector<double> beta_i(n);
    for (int j = 0; j < n; ++j)
      beta_i[j] = beta[layernorm_feature(i, j, n, row, batch_size)];
    PhantomPlaintext ecd_beta;
    encoder.encode(beta_i,x1x2[i].params_id(),x1x2[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(x1x2[i],ecd_beta);
  }

  return x1x2;

}

inline vector<PhantomCiphertext> layernorm_boot_final(vector<PhantomCiphertext> & x1, vector<PhantomCiphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta, double c,
   int row, int col, int batch_size,
  PhantomContext & seal_context, const PhantomRelinKey &relin_keys, PhantomGaloisKey & RotK, Bootstrapper& bootstrapper){
  
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);

  validate_layernorm_shape(x1.size(), x2.size(), encoder.slot_count(),
      row, col, batch_size, gamma, beta, c);

  double scale = x1[0].scale();
  size_t slot_count = encoder.slot_count();
  int n = slot_count;
  int num_ct = x1.size();

  if(seal_context.get_context_data(x1[0].params_id()).chain_depth() != 
    seal_context.get_context_data(x2[0].params_id()).chain_depth()){
    
  }
  if(x1.size() != x2.size()){
    
  }

  vector<PhantomCiphertext> x1x2(num_ct);
  for (int i = 0; i < num_ct; ++i){
    evaluator.add(x1[i],x2[i],x1x2[i]);
  }

  PhantomCiphertext sum = x1x2[0];
  for (int i = 1; i < num_ct; ++i){
    evaluator.add_inplace(sum,x1x2[i]);
  }

  int sum_index = n/2;
  int min_index = row * batch_size;
  while(sum_index >= min_index){
    PhantomCiphertext copy = sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_col(n, 1.0/(double)col);
  PhantomPlaintext ecd_inv_col;
  encoder.encode(inv_col, sum.params_id(), sum.scale(), ecd_inv_col);
  PhantomCiphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  PhantomCiphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);           
    PhantomCiphertext tmp;
    evaluator.square(x1x2[i],tmp);

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
    PhantomCiphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);   
    sum_index /= 2;
  }

  PhantomCiphertext rtn;
  bootstrapper.bootstrap_3(rtn,var_sum);   
  var_sum=rtn;

  evaluator.mod_switch_to_next_inplace(var_sum);  

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
  PhantomPlaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.params_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum);   

  PhantomCiphertext inv_sqrt_var = invert_sqrt_boot2(var_sum,4,2,seal_context,relin_keys,bootstrapper);  

  for (int i = 0; i < num_ct; ++i){

    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    PhantomPlaintext ecd_gamma;

    PhantomCiphertext scaled_inverse = inv_sqrt_var;
    encoder.encode(gamma_c,scaled_inverse.params_id(),scaled_inverse.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(scaled_inverse,ecd_gamma);
    evaluator.rescale_to_next_inplace(scaled_inverse);

    evaluator.mod_switch_to_inplace(x1x2[i], scaled_inverse.params_id());
    evaluator.multiply_inplace(x1x2[i],scaled_inverse);
    evaluator.relinearize_inplace(x1x2[i],relin_keys);
    evaluator.rescale_to_next_inplace(x1x2[i]);  

    vector<double> beta_i(n);
    for (int j = 0; j < n; ++j)
      beta_i[j] = beta[layernorm_feature(i, j, n, row, batch_size)];
    PhantomPlaintext ecd_beta;
    encoder.encode(beta_i,x1x2[i].params_id(),x1x2[i].scale(),ecd_beta);
    evaluator.add_plain_inplace(x1x2[i],ecd_beta);
  }

  return x1x2;

}
