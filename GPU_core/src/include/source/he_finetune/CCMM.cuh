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

inline void interal_rot_inplace(PhantomCiphertext & x, int rot_index, int vec_size, 
  int batch_size, int k, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();
  int vec_num = n/(vec_size*batch_size);

  PhantomCiphertext copy = x;

  vector<double> mask(n,0.0);
  for (int i = 0; i < vec_num; ++i){
    for (int j = 0 ; j < rot_index*batch_size ; ++j){
      mask[i*vec_size*batch_size+j] = 1.0;
    }
  }
  PhantomPlaintext ecd_mask;
  encoder.encode(mask, x.params_id(), x.scale(), ecd_mask);

  evaluator.multiply_plain_inplace(x,ecd_mask);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(copy, x.params_id());

  x.scale()=copy.scale();

  evaluator.sub_inplace(copy,x);

  evaluator.rotate_vector_inplace(copy, rot_index*batch_size, RotK);
 
  evaluator.rotate_vector_inplace(x, -1*(vec_size-rot_index)*batch_size, RotK);

  evaluator.add_inplace(x,copy);

  return;

}

inline vector<PhantomCiphertext> CCMM_LoRA_A(vector<PhantomCiphertext> & enc_X, 
  vector<PhantomCiphertext> & W, 
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  if(col_X != row_W){
    
  }

  if(enc_X.size()*col_W != W.size()){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int mid = row_X*col_X/n;
  int ed = mid * col_W;
  
  vector<PhantomCiphertext> output(ed);

  for (int i = 0; i < col_W; ++i){
    
    for (int j = 0; j < mid; ++j){
      evaluator.multiply(enc_X[j], W[i*mid+j], output[i*mid+j]);
      evaluator.relinearize_inplace(output[i*mid+j],relin_keys);
      evaluator.rescale_to_next_inplace(output[i*mid+j]);

      int rot_index = n/2;
      while(rot_index >= row_X){
        
        PhantomCiphertext tmp;
        evaluator.rotate_vector(output[i*mid+j], rot_index, RotK, tmp);
        evaluator.add_inplace(output[i*mid+j],tmp);
        rot_index /= 2;
      }
      if(j > 0){
        evaluator.add_inplace(output[i*mid],output[i*mid+j]);
      }
    }

    for (int j = 1 ; j < mid ; ++j){
      output[i*mid+j] = output[i*mid];
    }
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_LoRA_A_batch(vector<PhantomCiphertext> & enc_X, 
  vector<PhantomCiphertext> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  if(col_X != row_W){
    
  }

  if(enc_X.size()*col_W != W.size()){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int mid = row_X*col_X*batch_size/n;
  int ed = col_W;
  
  vector<PhantomCiphertext> output(ed);

  for (int i = 0; i < col_W; ++i){
    
    for (int j = 0; j < mid; ++j){
      PhantomCiphertext xa;
      evaluator.multiply(enc_X[j], W[i*mid+j], xa);

      if(j ==0){
        output[i] = xa;
      }
      else if(j > 0){
        evaluator.add_inplace(output[i],xa);
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

  return output;

}

inline vector<PhantomCiphertext> CCMM_LoRA_B_batch(vector<PhantomCiphertext> & enc_X, 
  vector<PhantomCiphertext> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  int mul_num = 0;

  if(col_X != row_W){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int mid = enc_X.size();
  int ed = row_X*col_W*batch_size/n;
  
  vector<PhantomCiphertext> output(ed);

  for (int i = 0; i < ed; ++i){
    for (int j = 0 ; j < mid ; ++j){
      PhantomCiphertext tmp;
      evaluator.multiply(enc_X[j],W[j*ed+i],tmp);
      mul_num++;

      if(j == 0){
        output[i] = tmp;
      }
      else{
        evaluator.add_inplace(output[i],tmp);
      }
    }
    evaluator.relinearize_inplace(output[i],relin_keys);     
    
    evaluator.rescale_to_next_inplace(output[i]);            
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_LoRA_B_batch_complex(vector<PhantomCiphertext> & enc_X, 
  vector<PhantomCiphertext> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  int mul_num = 0;

  if(col_X != row_W){
    
  }

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int mid = enc_X.size();
  int ed = row_X*col_W*batch_size/(2*n);
  
  vector<PhantomCiphertext> output(ed);

  for (int i = 0; i < ed; ++i){
    for (int j = 0 ; j < mid ; ++j){
      PhantomCiphertext tmp;
      evaluator.multiply(enc_X[j],W[j*ed+i],tmp);
      mul_num++;

      if(j == 0){
        output[i] = tmp;
      }
      else{
        evaluator.add_inplace(output[i],tmp);
      }
    }
    evaluator.relinearize_inplace(output[i],relin_keys);     
    
    evaluator.rescale_to_next_inplace(output[i]);            
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_Kernel_batch(vector<PhantomCiphertext> & enc_Q, 
  vector<PhantomCiphertext> & enc_K, int batch_size, int col, int row, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = enc_Q.size();
  int ed = row*row*batch_size/n;
  int num_diag = row/ed;
  
  vector<PhantomCiphertext> output(ed);

  for (int ind = 0 ; ind < ed ; ++ind){
    for (int i = 0; i < num_diag; ++i){
      int index = ind*num_diag+i;
      PhantomCiphertext tmpoutput;
      for (int j = 0; j < mid; ++j){
        PhantomCiphertext tmp=enc_K[j];
        interal_rot_inplace(tmp,index,row,batch_size,k,seal_context,RotK);
        
        pcm_num ++;
        PhantomCiphertext tmpQK;
        evaluator.sub(enc_Q[j],tmp,tmpQK); 
        evaluator.square_inplace(tmpQK);
        ccm_num++;

        if(j == 0){
          tmpoutput = tmpQK;
        }
        else{
          evaluator.add_inplace(tmpoutput,tmpQK);
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

      vector<double> mask(n,0.0);
      for(int j = i*row*batch_size ; j < (i+1)*row*batch_size ; ++j){
        mask[j] = 1.0;
      }
      PhantomPlaintext ecd_mask;
      encoder.encode(mask, tmpoutput.params_id(), tmpoutput.scale(), ecd_mask);
      evaluator.multiply_plain_inplace(tmpoutput,ecd_mask);
      pcm_num++;
      evaluator.rescale_to_next_inplace(tmpoutput);

      if(i == 0){
        output[ind] = tmpoutput;
      }
      else{
        evaluator.add_inplace(output[ind], tmpoutput);
      }
    }
    
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_Kernel_batch_bsgs(vector<PhantomCiphertext> & enc_Q, 
  vector<PhantomCiphertext> & enc_K, int batch_size, int col, int row, int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = enc_Q.size();
  int ed = row*row*batch_size/n;
  int num_diag = row/ed;

  vector<PhantomCiphertext> rot_Q(ed*mid);
  for(int ind = 0 ; ind < ed ; ++ind){
    for(int j = 0 ; j < mid ; ++j){
      rot_Q[ind*mid+j] = enc_Q[j];
      interal_rot_inplace(rot_Q[ind*mid+j],(row-(ind*row/ed)),row,batch_size,k,seal_context,RotK);
      
      pcm_num ++;
    }
  }

  vector<PhantomCiphertext> rot_K(num_diag*mid);
  for (int i = 0; i < num_diag; ++i){
    for (int j = 0 ; j < mid ; ++j){
      rot_K[i*mid+j] = enc_K[j];
      interal_rot_inplace(rot_K[i*mid+j],i,row,batch_size,k,seal_context,RotK);
      
      pcm_num ++;
    }
  }

  vector<PhantomCiphertext> output(ed);

  for (int ind = 0 ; ind < ed ; ++ind){
    for (int i = 0; i < num_diag; ++i){
      PhantomCiphertext tmpoutput;
      for (int j = 0; j < mid; ++j){
        PhantomCiphertext tmpQK;
        evaluator.sub(rot_Q[ind*mid+j],rot_K[i*mid+j],tmpQK); 
        evaluator.square_inplace(tmpQK);
        ccm_num++;

        if(j == 0){
          tmpoutput = tmpQK;
        }
        else{
          evaluator.add_inplace(tmpoutput,tmpQK);
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
      
      vector<double> mask(n,0.0);
      for(int j = i*row*batch_size ; j < (i+1)*row*batch_size ; ++j){
        mask[j] = 1.0;
      }
      PhantomPlaintext ecd_mask;
      encoder.encode(mask, tmpoutput.params_id(), tmpoutput.scale(), ecd_mask);
      evaluator.multiply_plain_inplace(tmpoutput,ecd_mask);
      pcm_num++;
      evaluator.rescale_to_next_inplace(tmpoutput);

      if(i == 0){
        output[ind] = tmpoutput;
      }
      else{
        evaluator.add_inplace(output[ind], tmpoutput);
      }
    }
    
  }

  return output;

}

inline vector<PhantomCiphertext> ext_QK_batch(vector<PhantomCiphertext> & enc_QK, int batch_size, int QK_row, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int ccm_num = 0;
  int pcm_num = 0;

  int midqk = enc_QK.size();
  int num_diag = QK_row/midqk;

  vector<PhantomCiphertext> output(QK_row);

    for (int i = 0 ; i < midqk ; ++i){
      for(int j = 0 ; j < num_diag ; ++j){
        
        output[i*num_diag+j] = enc_QK[i];
        vector<double> mask(n,0.0);
        for(int ind2 = j*QK_row*batch_size; ind2 < (j+1)*QK_row*batch_size; ++ind2){
          mask[ind2] = 1.0;
        }
        PhantomPlaintext ecd_mask;
        encoder.encode(mask, output[i*num_diag+j].params_id(), output[i*num_diag+j].scale(), ecd_mask);
        evaluator.multiply_plain_inplace(output[i*num_diag+j],ecd_mask);
        pcm_num++;
        evaluator.rescale_to_next_inplace(output[i*num_diag+j]);

        int sum_index = n/2;
        int min_index = QK_row * batch_size;
        while(sum_index >= min_index){
          PhantomCiphertext copy = output[i*num_diag+j];
          
          evaluator.rotate_vector_inplace(copy,sum_index,RotK);
          
          evaluator.add_inplace(output[i*num_diag+j],copy);
          sum_index /= 2;
        }
      }
    }

  return output;

}

inline vector<PhantomCiphertext> CCMM_V_batch(vector<PhantomCiphertext> & ext_QK, vector<PhantomCiphertext> & enc_V, 
  int batch_size, int QK_row, int V_col, int V_row,
  int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int ccm_num = 0;
  int pcm_num = 0;

  int midqk = enc_V.size()*2;
  int midv = enc_V.size();
  int ed = QK_row*V_col*batch_size/n;
  int num_diag = QK_row/midqk;
  
  vector<PhantomCiphertext> output(ed);

  for (int ind = 0; ind < midv; ++ind){

    for (int i = 0 ; i < midqk ; ++i){
      for(int j = 0 ; j < num_diag ; ++j){
        
        PhantomCiphertext tmp = ext_QK[i*num_diag+j];

        PhantomCiphertext tmp2 = enc_V[ind];
        int rot_index2 = i*num_diag+j;
        interal_rot_inplace(tmp2,rot_index2,V_row,batch_size,k,seal_context,RotK);
        
        pcm_num++;

        evaluator.multiply_inplace(tmp,tmp2);
        ccm_num++;

        if(i == 0 && j == 0){
          output[ind] = tmp;
        }
        else{
          evaluator.add_inplace(output[ind],tmp);
        }
      }
    }
    evaluator.relinearize_inplace(output[ind],relin_keys);       
    
    evaluator.rescale_to_next_inplace(output[ind]);         
  }

  return output;

}

inline vector<PhantomCiphertext> CCMM_V_batch_bsgs_complex(vector<PhantomCiphertext> & ext_QK, vector<PhantomCiphertext> & enc_V, 
  int batch_size, int QK_row, int V_col, int V_row,
  int k, 
  PhantomContext & seal_context, 
  PhantomGaloisKey & RotK, const PhantomRelinKey &relin_keys){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();

  int ccm_num = 0;
  int pcm_num = 0;

  int midqk = enc_V.size()*4;
  int midv = enc_V.size();
  int ed = QK_row*V_col*batch_size/n;
  int num_diag = QK_row/midqk;
  
  vector<PhantomCiphertext> output(ed);

  for (int ind = 0; ind < midv; ++ind){
    vector<PhantomCiphertext> tmpv(num_diag, enc_V[ind]);
    
    for (int j = 0 ; j < num_diag ; ++j){
        interal_rot_inplace(tmpv[j],j,V_row,batch_size,k,seal_context,RotK);
        
        pcm_num++;
    }

    for (int i = 0 ; i < midqk ; ++i){
      PhantomCiphertext tmpqkv;
      for(int j = 0 ; j < num_diag ; ++j){
        PhantomCiphertext tmp = ext_QK[i*num_diag+j];

        evaluator.multiply_inplace(tmp,tmpv[j]);
        ccm_num++;

        if(j == 0){
          tmpqkv = tmp;
        }
        else{
          evaluator.add_inplace(tmpqkv,tmp);
        }
      }

      evaluator.relinearize_inplace(tmpqkv,relin_keys);   
      
      evaluator.rescale_to_next_inplace(tmpqkv);

      interal_rot_inplace(tmpqkv,(i*QK_row/midqk),QK_row,batch_size,k,seal_context,RotK);
      
      pcm_num++;

      if(i == 0){
        output[ind*2] = tmpqkv;
      }
      else{
          evaluator.add_inplace(output[2*ind],tmpqkv);
        }
    }

    evaluator.complex_conjugate(output[2*ind], RotK, output[2*ind+1]);
    
    PhantomCiphertext tmpoutput;
    evaluator.add(output[2*ind],output[2*ind+1],tmpoutput);
    evaluator.sub(output[2*ind], output[2*ind+1], output[2*ind+1]);
    output[2*ind] = tmpoutput;
  }

  return output;

}
