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

inline vector<PhantomCiphertext> gradient_A(vector<PhantomCiphertext> & enc_X,
 vector<PhantomCiphertext> & gradient_Q, vector<PhantomCiphertext> & B,
 int batch_size, int row, int col, int rank,
 PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<PhantomCiphertext> real_B = Complex_to_real(B, seal_context, RotK); 

  vector<PhantomCiphertext> QBT = CCMM_LoRA_A_batch(gradient_Q, real_B, batch_size,  
        col, row, rank, col, 1, seal_context,RotK,relin_keys);

  return CCMM_gradient_Lora(enc_X,QBT,batch_size,row,col,rank,seal_context,RotK,relin_keys);  

}

inline vector<PhantomCiphertext> gradient_B(vector<PhantomCiphertext> & XA,
 vector<PhantomCiphertext> & gradient_Q, 
 int batch_size, int row, int col, int rank,
 PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  if(XA.size() != 2){
    
  }

  PhantomCiphertext complex_XA = XA[0];
  PhantomCiphertext tmp;
  vector<complex<double>> mask(n,complex<double>{0,1});
  PhantomPlaintext ecd_mask;
  encoder.encode(mask, complex_XA.params_id(), complex_XA.scale(), ecd_mask);
  evaluator.multiply_plain(XA[1],ecd_mask,tmp);
  evaluator.rescale_to_next_inplace(tmp);              
  evaluator.add_inplace(complex_XA, tmp);

  vector<PhantomCiphertext> output(48);

  for (int i = 0; i < 48; ++i){
    evaluator.multiply(complex_XA,gradient_Q[i],output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);    

    int rot_index = row/2;
    while(rot_index >= 1){
      PhantomCiphertext tmp2 = output[i];
      interal_rot_inplace(tmp2,rot_index,row,batch_size,1,seal_context,RotK); 
      
      evaluator.add_inplace(output[i],tmp2);
      rot_index /= 2;
    }
  }

  vector<PhantomCiphertext> real_B = Complex_to_real(output, seal_context, RotK); 

  evaluator.mod_switch_to_inplace(ecd_mask,real_B[0].params_id());

  for (int i = 0; i < 48; ++i){
    output[i] = real_B[2*i];
    evaluator.multiply_plain_inplace(real_B[2*i+1],ecd_mask);   
    evaluator.rescale_to_next_inplace(real_B[2*i+1]);
    evaluator.mod_switch_to_inplace(output[i],real_B[2*i+1].params_id());
    evaluator.add_inplace(output[i],real_B[2*i+1]);
  }

  return output;

}

inline vector<PhantomCiphertext> gradient_x_part1(vector<PhantomCiphertext> gradient_Q, 
  vector<PhantomCiphertext> & A, vector<PhantomCiphertext> & B, const vector<vector<double>> W,
  int batch_size, int row, int col, int rank,
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  vector<vector<double>> WT=matrix_trans(W);

  vector<PhantomCiphertext> QWT = PCMM_modified_col_packing_batch(gradient_Q, WT, 
        batch_size, col, row, col, col, 1, seal_context,RotK);

  vector<PhantomCiphertext> real_B = Complex_to_real(B, seal_context, RotK); 

  vector<PhantomCiphertext> QBT = CCMM_LoRA_A_batch(gradient_Q, real_B, batch_size,
        col, row, rank, col, 1, seal_context,RotK,relin_keys);

  vector<PhantomCiphertext> QBTAT = CCMM_LoRA_B_batch(QBT,A,batch_size,
        rank,row,col,rank,1,seal_context,RotK,relin_keys);

  for (int i = 0; i < QWT.size(); ++i){
    evaluator.mod_switch_to_inplace(QWT[i],QBTAT[i].params_id());
    evaluator.add_inplace(QWT[i],QBTAT[i]);
  }

  return QWT;
}
