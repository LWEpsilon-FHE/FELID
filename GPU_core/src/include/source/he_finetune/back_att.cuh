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

inline vector<PhantomCiphertext> gradient_LN1(vector<PhantomCiphertext> & gradient_att,  
  vector<PhantomCiphertext> & LN_no_norm, PhantomCiphertext & inv_sqrt_var, 
  const vector<double> & gamma, int batch_size, int num_col, int num_row, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  return gradient_LN(gradient_att, LN_no_norm, inv_sqrt_var, gamma, batch_size, num_col, num_row, seal_context, RotK, relin_keys);
}

inline vector<PhantomCiphertext> gradient_cat(vector<PhantomCiphertext> & gradient_LN1,  
  const vector<vector<double>> & Wo, int batch_size, int num_col, int num_row, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<vector<double>> WoT = matrix_trans(Wo);
  
  vector<PhantomCiphertext> LNWoT = PCMM_modified_col_packing_batch(gradient_LN1, WoT, batch_size, 
        768, 128, 768, 768, 1, seal_context,RotK);

  if(LNWoT.size()!=48){
    
  }

  return LNWoT;

}

inline vector<PhantomCiphertext> gradient_U(vector<PhantomCiphertext> & gradient_cat, 
  vector<PhantomCiphertext> & V, int batch_size, int row, int col, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> rot_G=rot_G_bs(gradient_cat,batch_size,row,seal_context,RotK);

  return CCMM_GV_batch_bsgs(rot_G,V,batch_size,col,row,seal_context,RotK,relin_keys);

}

inline vector<PhantomCiphertext> compute_D(vector<PhantomCiphertext> & gradient_U, vector<PhantomCiphertext> & ext_QK, 
  int batch_size, int row, PhantomContext & seal_context, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> output(row);
  for (int i = 0; i < row; ++i){
    evaluator.multiply(gradient_U[i], ext_QK[i], output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);
  }

  return output;

}

inline void align_gradient_operands(PhantomCiphertext &a, PhantomCiphertext &b, Evaluator &evaluator) {
  if(a.coeff_modulus_size()>b.coeff_modulus_size()) evaluator.mod_switch_to_inplace(a,b.params_id());
  else if(b.coeff_modulus_size()>a.coeff_modulus_size()) evaluator.mod_switch_to_inplace(b,a.params_id());
  a.scale()=b.scale();
}

inline vector<PhantomCiphertext> gradient_Q(vector<PhantomCiphertext> & D, vector<PhantomCiphertext> & Q, vector<PhantomCiphertext> & rot_K,
  int batch_size, int row, int col, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> DK = CCMM_DK_batch_bsgs_complex(D,rot_K,batch_size,row,col,row,seal_context,RotK,relin_keys); 

  PhantomCiphertext sum_D;
  for (int i = 0; i < row/16; ++i){
    PhantomCiphertext tmp = D[i*16];
    for(int j = 1 ; j < 16 ; ++j){
      evaluator.add_inplace(tmp,D[i*16+j]);
    }
    
    evaluator.rotate_vector_inplace(tmp,16*i*batch_size,RotK);
    
    if(i == 0){
      sum_D = tmp;
    }
    else{
      evaluator.add_inplace(sum_D,tmp);
    }
  }

  vector<PhantomCiphertext> real_Q = Complex_to_real_back(Q, 2.0, seal_context,RotK); 

  vector<PhantomCiphertext> output(real_Q.size());
  for (int i = 0; i < real_Q.size(); ++i){
    PhantomCiphertext lhs=sum_D,value=real_Q[i];
    align_gradient_operands(lhs,value,evaluator);
    evaluator.multiply(lhs,value,output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys); 
    
    evaluator.rescale_to_next_inplace(output[i]);
    PhantomCiphertext product=DK[i];
    align_gradient_operands(output[i],product,evaluator);
    evaluator.sub_inplace(output[i],product);
  }

  return output;

}

inline vector<PhantomCiphertext> gradient_K(vector<PhantomCiphertext> & D, vector<PhantomCiphertext> & K, vector<PhantomCiphertext> & Q,
  int batch_size, int row, int col, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> DT=rot_UT_gradV(D,batch_size,row,seal_context,RotK);

  vector<PhantomCiphertext> DTQ = CCMM_V_batch(DT,Q,batch_size,
        row, col, row, 1, seal_context,RotK,relin_keys); 

  PhantomCiphertext sum_DT;
  for (int i = 0; i < row/16; ++i){
    PhantomCiphertext tmp = DT[i*16];
    for(int j = 1 ; j < 16 ; ++j){
      evaluator.add_inplace(tmp,DT[i*16+j]);
    }

    if(i == 0){
      sum_DT = tmp;
    }
    else{
      evaluator.add_inplace(sum_DT,tmp);
    }
  }

  vector<PhantomCiphertext> real_K = Complex_to_real_back(K, 2.0, seal_context,RotK); 

  vector<PhantomCiphertext> output(real_K.size());
  for (int i = 0; i < real_K.size(); ++i){
    PhantomCiphertext lhs=sum_DT,value=real_K[i];
    align_gradient_operands(lhs,value,evaluator);
    evaluator.multiply(lhs,value,output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);
    PhantomCiphertext product=DTQ[i];
    align_gradient_operands(output[i],product,evaluator);
    evaluator.sub_inplace(output[i],product);
  }

  return output;

}

inline vector<PhantomCiphertext> gradient_V(vector<PhantomCiphertext> & gradient_cat, 
  vector<PhantomCiphertext> & ext_U, int batch_size, int U_row, int num_col, int num_row,
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> UT = rot_UT_gradV(ext_U,batch_size,U_row,seal_context,RotK);

  return CCMM_V_batch(UT, gradient_cat, batch_size,
        U_row, num_col, num_row, 1, seal_context,RotK,relin_keys);

}
