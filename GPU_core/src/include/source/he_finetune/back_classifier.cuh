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

inline PhantomCiphertext partial_L_y(PhantomCiphertext & y_l, PhantomCiphertext & y_r, PhantomContext & seal_context){
  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  PhantomCiphertext yl_minus_yr;
  evaluator.sub(y_l,y_r,yl_minus_yr);
  return yl_minus_yr;

}

inline PhantomCiphertext back_class_2_batch(PhantomCiphertext & y_lr, PhantomCiphertext & tanhx, 
  int batch_size, int col_W, int k, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  PhantomCiphertext yl_minus_yr=y_lr;

  evaluator.multiply_inplace(yl_minus_yr,tanhx);
  evaluator.relinearize_inplace(yl_minus_yr,relin_keys);

  evaluator.rescale_to_next_inplace(yl_minus_yr);

  return yl_minus_yr;
}

inline vector<PhantomCiphertext> pre_compute_back_class_1(PhantomCiphertext & y_lr, PhantomCiphertext & W2, 
  PhantomCiphertext & tanhx, 
  int batch_size, int col_W, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  PhantomCiphertext tanh2;
  evaluator.square(tanhx,tanh2);
  
  evaluator.relinearize_inplace(tanh2, relin_keys);
  
  evaluator.rescale_to_next_inplace(tanh2);          

  evaluator.negate_inplace(tanh2);
  
  vector<double> one(n,1.0);
  PhantomPlaintext ecd1;
  encoder.encode(one, tanh2.params_id(), tanh2.scale(), ecd1);
  evaluator.add_plain_inplace(tanh2,ecd1);

  PhantomCiphertext output;

  evaluator.multiply(y_lr,W2,output);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);        

  PhantomCiphertext tmp;
  evaluator.rotate_vector(output, n/2, RotK, tmp);
  evaluator.add_inplace(output,tmp);

  evaluator.multiply_inplace(output,tanh2);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);

  vector<PhantomCiphertext> final_output(6); 
  for (int i = 0; i < 6; ++i){
    vector<double> mask(n,0);
    for (int j = i*128*batch_size ; j < (i+1)*128*batch_size ; ++j){
      mask[j] = 1.0;
      mask[j+n/2] = 1.0;
    }
    PhantomPlaintext ecd_mask;
    encoder.encode(mask, output.params_id(), output.scale(),ecd_mask);
    evaluator.multiply_plain(output, ecd_mask, final_output[i]);
    evaluator.rescale_to_next_inplace(final_output[i]);

    int rot_index = 128*batch_size;
    while(rot_index <= n/4){
      PhantomCiphertext tmp2;
      evaluator.rotate_vector(final_output[i], rot_index, RotK, tmp2);
      evaluator.add_inplace(final_output[i], tmp2);
      rot_index *= 2;
    }
  }

  return final_output;

}

inline vector<PhantomCiphertext> back_class_1_batch(vector<PhantomCiphertext> & ct, 
vector<PhantomCiphertext> & pre_compute_ct, 
  int batch_size, int col_W, int row_W, int k, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> ext_ct = ext_copy_X(ct,batch_size, 768,128,seal_context,RotK);  

  vector<PhantomCiphertext> output(ext_ct.size());
  for (int i = 0; i < 6; ++i){   
    for (int j = 0 ; j < 48 ; ++j){      
      evaluator.multiply(ext_ct[i*48+j], pre_compute_ct[i], output[i*48+j]);
      
      evaluator.relinearize_inplace(output[i*48+j],relin_keys);
      
      evaluator.rescale_to_next_inplace(output[i*48+j]);
    }
  }

  return output;

}

inline vector<PhantomCiphertext> gradient_x2(vector<PhantomCiphertext> & W1, vector<PhantomCiphertext> & pre_compute_ct, 
  int batch_size, int col_W, int row_W, int k, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> output(48); 

  vector<double> mask(n,0.0);
  for (int i = 0; i < 16; ++i){
    for (int j = 0 ; j < batch_size ; ++j){
      mask[i*128*batch_size+j] = 1.0;
    }
  }
  PhantomPlaintext ecd_mask;
  encoder.encode(mask, pre_compute_ct[0].params_id(), pre_compute_ct[0].scale(), ecd_mask);

  for (int i = 0; i < 48; ++i){
    
    for (int j = 0 ; j < 6 ; ++j){
      PhantomCiphertext tmp;
      evaluator.multiply(W1[j*48+i],pre_compute_ct[j],tmp);
      
      if (j == 0){
        output[i] = tmp;
      }
      else{
        evaluator.add_inplace(output[i],tmp);
      }
    }
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);

    int rot_index = batch_size;
    while(rot_index<128*batch_size){
      PhantomCiphertext tmp;
      evaluator.rotate_vector(output[i], rot_index, RotK, tmp);
      
      evaluator.add_inplace(output[i],tmp);
      rot_index *= 2;
    }

    evaluator.mod_switch_to_inplace(ecd_mask,output[i].params_id());
    evaluator.multiply_plain_inplace(output[i],ecd_mask);
    evaluator.rescale_to_next_inplace(output[i]);

  }

  return output;
}
