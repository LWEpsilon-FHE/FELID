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

inline vector<PhantomCiphertext> ext_copy_X(vector<PhantomCiphertext> & enc_X, 
  int batch_size, int col_X, int row_X,
  PhantomContext & seal_context, PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<double> mask(n,0);
  for (int i = 0; i < n/(row_X*batch_size); ++i){
    for (int j = 0 ; j < batch_size ; ++j){
      mask[i*row_X*batch_size+j] = 1;
    }
  }

  PhantomPlaintext ecd_mask;
  encoder.encode(mask, enc_X[0].params_id(), enc_X[0].scale(), ecd_mask);

  vector<PhantomCiphertext> ext_X(enc_X.size());
  for (int i = 0; i < enc_X.size(); ++i){
    evaluator.multiply_plain(enc_X[i],ecd_mask,ext_X[i]);
    evaluator.rescale_to_next_inplace(ext_X[i]);             
    
    int rot_index = batch_size;
    while(rot_index < row_X*batch_size){
      PhantomCiphertext tmp;
      evaluator.rotate_vector(ext_X[i],-rot_index,RotK,tmp);
      
      evaluator.add_inplace(ext_X[i],tmp);
      rot_index *= 2;
    }
  }

  return ext_X;

}

inline PhantomCiphertext CCMM_class_1_batch(vector<PhantomCiphertext> & ext_X, 
  vector<PhantomCiphertext> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  if(col_X != row_W){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int mid = row_X*col_X*batch_size/n;
  int ed = W.size()/mid;

  vector<PhantomCiphertext> output(ed);

  for (int i = 0; i < ed; ++i){
    
    for (int j = 0 ; j < mid ; ++j){
     
      PhantomCiphertext tmp;
      evaluator.multiply(ext_X[j],W[i*mid+j],tmp);     

      if(j == 0){
        output[i] = tmp;
      }
      else{
        evaluator.add_inplace(output[i],tmp);
      }
    }
   
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);

    int rot_index = n/2;
    while(rot_index >= (row_X*batch_size)){
     
      PhantomCiphertext tmp;
      evaluator.rotate_vector(output[i], rot_index, RotK, tmp);
      
      evaluator.add_inplace(output[i],tmp);
      rot_index /= 2;
    }
  }

  PhantomCiphertext final_output;
  
  for (int i = 0; i < ed; ++i){
    
    vector<double> mask(n,0.0);
    for(int j = i*row_X*batch_size ; j < (i+1)*row_X*batch_size ; ++j){
      mask[j] = 1.0;
    }
    PhantomPlaintext ecd_mask;
    encoder.encode(mask, output[i].params_id(), output[i].scale(), ecd_mask);
    evaluator.multiply_plain_inplace(output[i],ecd_mask);
    evaluator.rescale_to_next_inplace(output[i]);  

    if(i == 0){
      final_output = output[i];
    }
    else{
      evaluator.add_inplace(final_output,output[i]);
    }
  }

  PhantomCiphertext tmp;
  evaluator.rotate_vector(final_output,n/2,RotK,tmp);
  
  evaluator.add_inplace(final_output,tmp);

  return final_output;

}

inline PhantomCiphertext CCMM_class_2_batch(PhantomCiphertext & enc_X, 
  PhantomCiphertext & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  if(col_X != row_W){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  PhantomCiphertext output = enc_X;
  
  evaluator.multiply_inplace(output, W);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);

  int rot_index = n/4;
  while(rot_index >= batch_size){
    PhantomCiphertext tmp;
    evaluator.rotate_vector(output,rot_index,RotK,tmp);
    
    evaluator.add_inplace(output,tmp);
    rot_index /= 2;
  }

  return output;
}
