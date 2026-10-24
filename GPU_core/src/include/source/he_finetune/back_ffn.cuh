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

inline vector<vector<double>> matrix_trans(const vector<vector<double>> & input){
  vector<vector<double>> output(input[0].size(), vector<double>(input.size(),0));
  for (int i = 0; i < input[0].size(); ++i){
    for (int j = 0 ; j < input.size() ; ++j){
      output[i][j] = input[j][i];
    }
  }
  return output;
}

inline vector<PhantomCiphertext> gradient_LN( vector<PhantomCiphertext> & gradient_x,
   vector<PhantomCiphertext> & normalized_x,  PhantomCiphertext & inv_sqrt_var,
  const vector<double> & gamma, int batch_size, int num_col, int num_row,
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey & relin_keys,
  bool input_is_normalized = true){
  PhantomCKKSEncoder phantom_encoder(seal_context); Encoder encoder(&seal_context,&phantom_encoder);
  Evaluator evaluator(&seal_context,&phantom_encoder);
  const int n=encoder.slot_count();
  const int count=gradient_x.size(),features=n/(batch_size*num_row);
  if(count<=0 || count!=normalized_x.size() || count*features!=num_col || gamma.size()!=num_col)
    throw std::invalid_argument("gradient_LN packing shape");
  auto align=[&](PhantomCiphertext &a, PhantomCiphertext &b){
    if(a.coeff_modulus_size()>b.coeff_modulus_size()) evaluator.mod_switch_to_inplace(a,b.params_id());
    else if(b.coeff_modulus_size()>a.coeff_modulus_size()) evaluator.mod_switch_to_inplace(b,a.params_id());
    a.scale()=b.scale();
  };
  auto reduce=[&](PhantomCiphertext value){
    for(int step=n/2;step>=num_row*batch_size;step/=2){
      PhantomCiphertext tmp;evaluator.rotate_vector(value,step,RotK,tmp);evaluator.add_inplace(value,tmp);
    }
    return value;
  };
  vector<PhantomCiphertext> h(count);
  for(int i=0;i<count;++i){
    vector<double> slots(n);
    for(int j=0;j<n;++j) slots[j]=gamma[i*features+j/(num_row*batch_size)]/num_col;
    PhantomPlaintext coefficients;encoder.encode(slots,gradient_x[i].params_id(),gradient_x[i].scale(),coefficients);
    evaluator.multiply_plain(gradient_x[i],coefficients,h[i]);evaluator.rescale_to_next_inplace(h[i]);
  }
  PhantomCiphertext mean=h.front();
  for(int i=1;i<count;++i)evaluator.add_inplace(mean,h[i]);
  mean=reduce(mean);
  PhantomCiphertext projection;
  for(int i=0;i<count;++i){
    PhantomCiphertext z=normalized_x[i],g=h[i],product;align(z,g);evaluator.multiply(z,g,product);
    if(i==0)projection=product;else evaluator.add_inplace(projection,product);
  }
  evaluator.relinearize_inplace(projection,relin_keys);evaluator.rescale_to_next_inplace(projection);
  projection=reduce(projection);
  if(!input_is_normalized){
    PhantomCiphertext inverse2=inv_sqrt_var;evaluator.square_inplace(inverse2);
    evaluator.relinearize_inplace(inverse2,relin_keys);evaluator.rescale_to_next_inplace(inverse2);
    align(projection,inverse2);evaluator.multiply_inplace(projection,inverse2);
    evaluator.relinearize_inplace(projection,relin_keys);evaluator.rescale_to_next_inplace(projection);
  }
  for(int i=0;i<count;++i){
    PhantomCiphertext z=normalized_x[i],p=projection;align(z,p);
    evaluator.multiply_inplace(z,p);evaluator.relinearize_inplace(z,relin_keys);evaluator.rescale_to_next_inplace(z);
    PhantomPlaintext hidden_size;encoder.encode(vector<double>(n,num_col),h[i].params_id(),h[i].scale(),hidden_size);
    evaluator.multiply_plain_inplace(h[i],hidden_size);evaluator.rescale_to_next_inplace(h[i]);
    PhantomCiphertext m=mean;align(h[i],m);evaluator.sub_inplace(h[i],m);align(h[i],z);evaluator.sub_inplace(h[i],z);
    PhantomCiphertext inv=inv_sqrt_var;align(h[i],inv);evaluator.multiply_inplace(h[i],inv);
    evaluator.relinearize_inplace(h[i],relin_keys);evaluator.rescale_to_next_inplace(h[i]);
  }
  return h;
}

inline vector<PhantomCiphertext> gradient_att(vector<PhantomCiphertext> & gradient_LN, 
  vector<PhantomCiphertext> & tanh_gelu, vector<PhantomCiphertext> & gelu_input,  
  const vector<vector<double>> & wf1, const vector<vector<double>> & wf2,
  int batch_size, int num_col, int num_row, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<vector<double>> wf2T = matrix_trans(wf2);
  
  vector<PhantomCiphertext> Gzwf2T = PCMM_modified_col_packing_batch(gradient_LN, wf2T, batch_size,   
        768, 128, 3072, 768, 1, seal_context,RotK);

  for (int i = 0; i < 48*4; ++i){

    PhantomCiphertext tanh2;
    evaluator.square(tanh_gelu[i],tanh2);
    
    evaluator.relinearize_inplace(tanh2, relin_keys);
    
    evaluator.rescale_to_next_inplace(tanh2);          

    evaluator.negate_inplace(tanh2);
    
    vector<double> one(n,1.0);
    PhantomPlaintext ecd1;
    encoder.encode(one, tanh2.params_id(), tanh2.scale(), ecd1);
    evaluator.add_plain_inplace(tanh2,ecd1);

    PhantomCiphertext x2;
    evaluator.square(gelu_input[i],x2);
    
    evaluator.relinearize_inplace(x2,relin_keys);  
    
    evaluator.rescale_to_next_inplace(x2);
    vector<double> c(n,0.134145*0.5*(sqrt(2.0/3.1415926)));
    encoder.encode(c,x2.params_id(),x2.scale(),ecd1);
    evaluator.multiply_plain_inplace(x2,ecd1);
    evaluator.rescale_to_next_inplace(x2);     
    vector<double> c2(n,0.5*sqrt(2.0/3.1415926));
    encoder.encode(c2,x2.params_id(),x2.scale(),ecd1);
    evaluator.add_plain_inplace(x2,ecd1);

    evaluator.mod_switch_to_inplace(tanh2,x2.params_id());
    evaluator.multiply_inplace(tanh2,x2);
    
    evaluator.relinearize_inplace(tanh2,relin_keys);
    
    evaluator.rescale_to_next_inplace(tanh2);  

    vector<double> c3(n,0.5);
    encoder.encode(c3,tanh_gelu[i].params_id(),tanh_gelu[i].scale(),ecd1);
    PhantomCiphertext tmp;
    evaluator.multiply_plain(tanh_gelu[i],ecd1,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    evaluator.add_plain_inplace(tmp,ecd1);

    evaluator.mod_switch_to_inplace(tmp,tanh2.params_id());
    evaluator.add_inplace(tanh2,tmp);

    evaluator.multiply_inplace(Gzwf2T[i],tanh2);
    
    evaluator.relinearize_inplace(Gzwf2T[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(Gzwf2T[i]);

  }

  vector<vector<double>> wf1T = matrix_trans(wf1);
  
  vector<PhantomCiphertext> Gzwf1T = PCMM_modified_col_packing_batch(Gzwf2T, wf1T, batch_size,   
        3072, 128, 768, 3072, 1, seal_context,RotK);

  if(Gzwf1T.size() != 48){
    
  }

  for (int i = 0; i < 48; ++i){
    PhantomCiphertext tmp = gradient_LN[i];
    evaluator.mod_switch_to_inplace(tmp, Gzwf1T[i].params_id());
    evaluator.add_inplace(tmp,Gzwf1T[i]);
  }

  return Gzwf1T;

}
