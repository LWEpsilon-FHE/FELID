#pragma once
#include "Layernorm_affine.hpp"
#include "Bootstrapper.h"

using namespace std;
using namespace seal;

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

inline Ciphertext evalLine(Ciphertext x, Plaintext m, Plaintext c, const SEALContext& seal_context){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double scale = x.scale();

  evaluator.mod_switch_to_inplace(m,x.parms_id());
  evaluator.multiply_plain_inplace(x,m);
  evaluator.rescale_to_next_inplace(x);
  evaluator.mod_switch_to_inplace(c,x.parms_id());
  x.scale() = scale;
  evaluator.add_plain_inplace(x,c);
  return x;
}

inline Ciphertext initGuess(Ciphertext x, const SEALContext& seal_context){
  CKKSEncoder encoder(seal_context);
  Plaintext a,b;
  encoder.encode(-1.29054537e-04, x.scale(), a);
  encoder.encode(1.29054537e-01, x.scale(), b);
  return evalLine(x,a,b,seal_context);
}

inline Ciphertext newtonIter(Ciphertext x, Ciphertext res, int iter, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  double scale = x.scale();

  for (int i = 0; i < iter; ++i){
    Plaintext three_half, neg_half;
    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    Ciphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq,relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    Ciphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half,x.parms_id());
    evaluator.multiply_plain(x,neg_half,res_x);
    evaluator.rescale_to_next_inplace(res_x);
    if(seal_context.get_context_data(res.parms_id())->chain_index()<
      seal_context.get_context_data(res_x.parms_id())->chain_index()){
      evaluator.mod_switch_to_inplace(res_x,res.parms_id());
    }
    else{
      evaluator.mod_switch_to_inplace(res,res_x.parms_id());
    }

    evaluator.multiply_inplace(res_x,res);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(res_sq,res_x.parms_id());
    evaluator.multiply_inplace(res_x,res_sq);
    evaluator.relinearize_inplace(res_x,relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(three_half, res.parms_id());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    evaluator.mod_switch_to_inplace(res, res_x.parms_id());
    res_x.scale()=scale;
    res.scale()=scale;
    evaluator.add_inplace(res, res_x);
  }
  return res;
}

inline Ciphertext goldSchmidtIter(Ciphertext v, Ciphertext y, int d, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  double scale = y.scale();

  Plaintext constant;
  encoder.encode(0.5,scale,constant);

  evaluator.mod_switch_to_inplace(v,y.parms_id());
  Ciphertext x;
  evaluator.multiply(v,y,x);
  evaluator.relinearize_inplace(x,relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant,y.parms_id());
  Ciphertext h;
  evaluator.multiply_plain(y,constant,h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; ++i){
    encoder.encode(0.5,scale,constant);
    Ciphertext r;
    evaluator.multiply(x,h,r);
    evaluator.relinearize_inplace(r,relin_keys);
    evaluator.rescale_to_next_inplace(r);
    r.scale() = scale;

    Ciphertext temp;
    evaluator.negate(r,temp);
    evaluator.mod_switch_to_inplace(constant,temp.parms_id());
    evaluator.add_plain(temp, constant, r);

    evaluator.mod_switch_to_inplace(x,r.parms_id());
    evaluator.multiply(x,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    x.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(x,temp.parms_id());
    evaluator.add_inplace(x,temp);

    evaluator.mod_switch_to_inplace(h,r.parms_id());
    evaluator.multiply(h,r,temp);
    evaluator.relinearize_inplace(temp,relin_keys);
    evaluator.rescale_to_next_inplace(temp);
    h.scale()=scale;
    temp.scale()=scale;
    evaluator.mod_switch_to_inplace(h,temp.parms_id());
    evaluator.add_inplace(h,temp);
  }
  encoder.encode(2.0,scale,constant);
  evaluator.mod_switch_to_inplace(constant, h.parms_id());
  evaluator.multiply_plain_inplace(h,constant);
  evaluator.rescale_to_next_inplace(h);

  return h;

}

inline Ciphertext invert_sqrt(Ciphertext x, int d_newt, int d_gold, 
  const SEALContext& seal_context, const RelinKeys &relin_keys){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Ciphertext res = initGuess(x, seal_context);    

  Ciphertext y = newtonIter(x,res,d_newt,seal_context,relin_keys);    

  Ciphertext sqrt_inv = goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline Ciphertext invert_sqrt_boot(Ciphertext x, int d_newt, int d_gold, 
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

inline Ciphertext invert_sqrt_boot2(Ciphertext x, int d_newt, int d_gold, 
  const SEALContext& seal_context, const RelinKeys &relin_keys, Bootstrapper& bootstrapper){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);

  Ciphertext res = initGuess(x, seal_context);    

  for (int j = 0 ; j < 1 ; ++j){
    evaluator.mod_switch_to_next_inplace(res);   
  }

  Ciphertext y = newtonIter(x,res,d_newt,seal_context,relin_keys);    

  while(seal_context.get_context_data(y.parms_id())->chain_index()>0){
  evaluator.mod_switch_to_next_inplace(y);
  }
  Ciphertext rtn;
  bootstrapper.bootstrap_3(rtn,y);   
  y=rtn;

  for (int j = 0 ; j < 4 ; ++j){
    evaluator.mod_switch_to_next_inplace(y);   
  }

  Ciphertext sqrt_inv = goldSchmidtIter(x,y,d_gold,seal_context,relin_keys); 
  return sqrt_inv;
}

inline vector<Ciphertext> layernorm(const vector<Ciphertext> & x1, const vector<Ciphertext> & x2, 
  const vector<double>& gamma, const vector<double>& beta, double c,
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

  vector<double> inv_col(n, 1.0/(double)col);
  Plaintext ecd_inv_col;
  encoder.encode(inv_col, sum.parms_id(), sum.scale(), ecd_inv_col);
  Ciphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  Ciphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);
    Ciphertext tmp;
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
    Ciphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
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

    evaluator.mod_switch_to_inplace(x1x2[i], inv_sqrt_var.parms_id());
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

inline vector<Ciphertext> layernorm_boot_self(const vector<Ciphertext> & x1, const vector<Ciphertext> & x2, 
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

  vector<double> inv_col(n, 1.0/(double)col);
  Plaintext ecd_inv_col;
  encoder.encode(inv_col, sum.parms_id(), sum.scale(), ecd_inv_col);
  Ciphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  Ciphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);           
    Ciphertext tmp;
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
    Ciphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);
    sum_index /= 2;
  }

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
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

    Ciphertext scaled_inverse = inv_sqrt_var;
    encoder.encode(gamma_c,scaled_inverse.parms_id(),scaled_inverse.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(scaled_inverse,ecd_gamma);
    evaluator.rescale_to_next_inplace(scaled_inverse);

    evaluator.mod_switch_to_inplace(x1x2[i], scaled_inverse.parms_id());
    evaluator.multiply_inplace(x1x2[i],scaled_inverse);
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

inline vector<Ciphertext> layernorm_boot_final(const vector<Ciphertext> & x1, const vector<Ciphertext> & x2, 
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

  vector<double> inv_col(n, 1.0/(double)col);
  Plaintext ecd_inv_col;
  encoder.encode(inv_col, sum.parms_id(), sum.scale(), ecd_inv_col);
  Ciphertext ave;
  evaluator.multiply_plain(sum, ecd_inv_col, ave);
  evaluator.rescale_to_next_inplace(ave);

  Ciphertext var_sum;
  for (int i = 0; i < num_ct; ++i){
    evaluator.mod_switch_to_next_inplace(x1x2[i]);    
    x1x2[i].scale()=ave.scale();
    evaluator.sub_inplace(x1x2[i],ave);           
    Ciphertext tmp;
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
    Ciphertext copy = var_sum;
    
    evaluator.rotate_vector_inplace(copy,sum_index,RotK);
    evaluator.add_inplace(var_sum,copy);   
    sum_index /= 2;
  }

  Ciphertext rtn;
  bootstrapper.bootstrap_3(rtn,var_sum);   
  var_sum=rtn;

  evaluator.mod_switch_to_next_inplace(var_sum);  

  vector<double> inv_cn(n, 1.0/((double)(col)*c));
  Plaintext ecd_inv_cn;
  encoder.encode(inv_cn, var_sum.parms_id(), var_sum.scale(), ecd_inv_cn);
  evaluator.multiply_plain_inplace(var_sum,ecd_inv_cn);
  evaluator.rescale_to_next_inplace(var_sum);   

  Ciphertext inv_sqrt_var = invert_sqrt_boot2(var_sum,4,2,seal_context,relin_keys,bootstrapper);  

  for (int i = 0; i < num_ct; ++i){

    vector<double> gamma_c(n,0);
    for (int j = 0 ; j < n ; ++j){
      gamma_c[j] = gamma[layernorm_feature(i, j, n, row, batch_size)]/sqrt(c);
    }
    Plaintext ecd_gamma;

    Ciphertext scaled_inverse = inv_sqrt_var;
    encoder.encode(gamma_c,scaled_inverse.parms_id(),scaled_inverse.scale(),ecd_gamma);
    evaluator.multiply_plain_inplace(scaled_inverse,ecd_gamma);
    evaluator.rescale_to_next_inplace(scaled_inverse);

    evaluator.mod_switch_to_inplace(x1x2[i], scaled_inverse.parms_id());
    evaluator.multiply_inplace(x1x2[i],scaled_inverse);
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
