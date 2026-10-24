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

inline vector<PhantomCiphertext> rot_G_bs(vector<PhantomCiphertext> & enc_G, 
  int batch_size, int row, PhantomContext & seal_context, 
  PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = enc_G.size();  
  int ed = row*row*batch_size/n;   
  int num_diag = row/ed;   

  vector<PhantomCiphertext> rot_G(ed*mid);
  for(int ind = 0 ; ind < ed ; ++ind){
    for(int j = 0 ; j < mid ; ++j){
      rot_G[ind*mid+j] = enc_G[j];
      interal_rot_inplace(rot_G[ind*mid+j],(row-(ind*row/ed)),row,batch_size,1,seal_context,RotK); 
      
      pcm_num ++;
    }
  }

  return rot_G;

}

inline vector<PhantomCiphertext> CCMM_GV_batch_bsgs(vector<PhantomCiphertext> & rot_G, vector<PhantomCiphertext> & enc_V, 
  int batch_size, int col, int row, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = 4;  
  int ed = row*row*batch_size/n;   
  int num_diag = row/ed;   

  vector<PhantomCiphertext> real_V = Complex_to_real_back(enc_V, (double)(-2*sqrt(128.0)), seal_context,RotK); 

  vector<PhantomCiphertext> rot_V(num_diag*mid);
  for (int i = 0; i < num_diag; ++i){
    for (int j = 0 ; j < mid ; ++j){
      rot_V[i*mid+j] = real_V[j];
      interal_rot_inplace(rot_V[i*mid+j],i,row,batch_size,1,seal_context,RotK); 
      
      pcm_num ++;
    }
  }

  vector<PhantomCiphertext> output(row);

  for (int ind = 0 ; ind < ed ; ++ind){
    for (int i = 0; i < num_diag; ++i){
      PhantomCiphertext tmpoutput;
      for (int j = 0; j < mid; ++j){
        PhantomCiphertext tmpGV;
        evaluator.multiply(rot_G[ind*mid+j],rot_V[i*mid+j],tmpGV);   
        ccm_num++;

        if(j == 0){
          tmpoutput = tmpGV;
        }
        else{
          evaluator.add_inplace(tmpoutput,tmpGV);
        }
      }
      evaluator.relinearize_inplace(tmpoutput,relin_keys);
        
        evaluator.rescale_to_next_inplace(tmpoutput);
      int sum_index = n/2;
      int min_index = row * batch_size;
      while(sum_index >= min_index){
          PhantomCiphertext copy = tmpoutput;
          
          evaluator.rotate_vector_inplace(copy,sum_index,RotK);
          
          evaluator.add_inplace(tmpoutput,copy);
          sum_index /= 2;
      }
      output[ind*num_diag+i] = tmpoutput;
    }
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_DK_batch_bsgs_complex(vector<PhantomCiphertext> & ext_D, vector<PhantomCiphertext> & rot_K, 
  int batch_size, int D_row, int K_col, int K_row,
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int ccm_num = 0;
  int pcm_num = 0;

  int midd = 8;
  int midk = 4;
  int ed = 4;
  int num_diag = 16;
  
  vector<PhantomCiphertext> output(ed);

  for (int ind = 0; ind < midk; ++ind){

    for (int i = 0 ; i < midd ; ++i){
      PhantomCiphertext tmpdk;
      for(int j = 0 ; j < num_diag ; ++j){
        PhantomCiphertext tmp = ext_D[i*num_diag+j];

        evaluator.multiply_inplace(tmp,rot_K[ind*16+j]);
        ccm_num++;

        if(j == 0){
          tmpdk = tmp;
        }
        else{
          evaluator.add_inplace(tmpdk,tmp);
        }
      }

      evaluator.relinearize_inplace(tmpdk,relin_keys);   
      
      evaluator.rescale_to_next_inplace(tmpdk);

      interal_rot_inplace(tmpdk,(i*D_row/midd),D_row,batch_size,1,seal_context,RotK);
      
      pcm_num++;

      if(i == 0){
        output[ind] = tmpdk;
      }
      else{
          evaluator.add_inplace(output[ind],tmpdk);
        }
    }
  }

  return output;

}

inline vector<PhantomCiphertext> rot_UT_gradV( vector<PhantomCiphertext> & ext_U,
        int batch_size, int U_row, 
        PhantomContext & seal_context, PhantomGaloisKey & RotK){

        PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
        int n = encoder.slot_count();
        vector<PhantomCiphertext> no_rot_UT(U_row);

        for (int i=0; i<U_row; ++i) {
            
            no_rot_UT[i] = ext_U[(U_row-i)%U_row];
            int U_idx = U_row - i;
            int left_rot_step = U_idx - (U_idx % (n/(U_row*batch_size))); 
            left_rot_step += i;                       
            
            while (left_rot_step <0) left_rot_step += U_row;
            evaluator.rotate_vector_inplace(no_rot_UT[i],left_rot_step*batch_size,RotK);         
            
        }

        return no_rot_UT;
}

inline vector<PhantomCiphertext> CCMM_gradient_Lora(vector<PhantomCiphertext> & enc_X, vector<PhantomCiphertext> & pre_lora_gradients,
  int batch_size, int row, int col, int rank,
  PhantomContext & seal_context, PhantomGaloisKey & RotK, const PhantomRelinKey& relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  if(enc_X.size()!=48){
    
  }
  if(pre_lora_gradients.size() != rank){
    
  }

  int num_ct = enc_X.size();
  vector<PhantomCiphertext> output(num_ct*rank);

  for (int i = 0; i < rank; ++i){
    for (int j = 0 ; j < num_ct ; ++j){
      
      evaluator.multiply(enc_X[j], pre_lora_gradients[i], output[i*num_ct+j]);
      
      evaluator.relinearize_inplace(output[i*num_ct+j],relin_keys);
      
      evaluator.rescale_to_next_inplace(output[i*num_ct+j]);
      
      int rot_index = row/2;
      while(rot_index >= 1){
        PhantomCiphertext tmp = output[i*num_ct+j];
        interal_rot_inplace(tmp,rot_index,row,batch_size,1,seal_context,RotK);
        
        evaluator.add_inplace(output[i*num_ct+j],tmp);
        rot_index /= 2;
      }
    }
  }

  return output;

}
