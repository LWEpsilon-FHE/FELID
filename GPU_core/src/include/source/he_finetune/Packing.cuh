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

inline vector<PhantomCiphertext> Modified_col_packing(const vector<vector<double>> & X, 
  int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){

  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();

  int num_ct = num_col*num_row/n;

  vector<PhantomCiphertext> output(num_ct);

  for (int i = 0; i < num_ct; ++i){
    vector<double> vec(n,0);
    for (int j = 0 ; j < n ; ++j){
      int row_index = j % num_row;
      int col_index = k* ((j/num_row) % (n/(k*num_row))) + (k*j)/n + (i)*(n/num_row);
      vec[j] = X[row_index][col_index];
    }
    PhantomPlaintext ecd_vec;
    encoder.encode(vec, 1, scale, ecd_vec);
    encryptor.encrypt(ecd_vec, output[i]);
  }

  return output;

}

inline vector<PhantomCiphertext> Modified_col_packing_batch(const vector<vector<vector<double>>> & X, 
  int batch_size, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){

  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();

  int num_ct = num_col*num_row*batch_size/n;

  vector<PhantomCiphertext> output(num_ct);
  int n2 = n/batch_size;

  for (int i = 0; i < num_ct; ++i){
    vector<double> vec(n,0);
    for (int j = 0 ; j < n2 ; ++j){
      for (int ind = 0 ; ind < batch_size ; ++ind){
        int row_index = j % num_row;
        int col_index = k* ((j/num_row) % (n2/(k*num_row))) + (k*j)/n2 + (i)*(n2/num_row);
        vec[j*batch_size+ind] = X[ind][row_index][col_index];
      }
    }
    PhantomPlaintext ecd_vec;
    encoder.encode(vec, 1, scale, ecd_vec);
    encryptor.encrypt(ecd_vec, output[i]);
  }
  
  return output;

}

inline vector<PhantomCiphertext> Modified_col_packing_batch_complex(const vector<vector<vector<double>>> & X, 
  int batch_size, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){

  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();

  int num_ct = num_col*num_row*batch_size/(2*n);

  vector<PhantomCiphertext> output(num_ct);
  int n2 = n/batch_size;

  for (int i = 0; i < num_ct; ++i){
    vector<complex<double>> vec(n,0);
    for (int j = 0 ; j < n2 ; ++j){
      for (int ind = 0 ; ind < batch_size ; ++ind){
        int row_index = j % num_row;
        int col_index = k* ((j/num_row) % (n2/(k*num_row))) + (k*j)/n2 + (2*i)*(n2/num_row);
        vec[j*batch_size+ind].real(X[ind][row_index][col_index]);
        int col_index2 = k* ((j/num_row) % (n2/(k*num_row))) + (k*j)/n2 + (2*i+1)*(n2/num_row);
        vec[j*batch_size+ind].imag(X[ind][row_index][col_index2]);
      }
    }
    PhantomPlaintext ecd_vec;
    encoder.encode(vec, 1, scale, ecd_vec);
    encryptor.encrypt(ecd_vec, output[i]);
  }
  
  return output;

}

inline vector<PhantomCiphertext> LoRA_A_packing_batch(const vector<vector<double>> & W, 
  int batch_size, int row_X, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();
  int num_ct_x = row_X*num_row*batch_size/n;
  
  int num_ct = num_col*num_ct_x;

  vector<PhantomCiphertext> output(num_ct);
  for (int i = 0; i < num_col; ++i){
    int n2 = num_row/num_ct_x; 
    
    for (int j = 0 ; j < num_ct_x ; ++j){
      vector<double> vec(n,0);
      for (int ind = 0 ; ind < n2 ; ++ind){
        for (int ind2 = 0 ; ind2 < (row_X*batch_size) ; ++ind2){
          vec[ind*row_X*batch_size+ind2] = W[j*n2+ind][i];
        }
      }
      
      PhantomPlaintext ecd_vec;
      encoder.encode(vec, 1, scale, ecd_vec);
      encryptor.encrypt(ecd_vec, output[i*num_ct_x+j]);
    }
  }

  return output;
}

inline vector<PhantomCiphertext> LoRA_B_packing_batch(const vector<vector<double>> & W, 
  int batch_size, int row_X, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();
  int num_ct_x = row_X*num_col*batch_size/n;
  
  int num_ct = num_row*num_ct_x;

  vector<PhantomCiphertext> output(num_ct);
  for (int i = 0; i < num_row; ++i){
    int n2 = num_col/num_ct_x; 
    
    for (int j = 0 ; j < num_ct_x ; ++j){
      vector<double> vec(n,0);
      for (int ind = 0 ; ind < n2 ; ++ind){
        for (int ind2 = 0 ; ind2 < (row_X*batch_size) ; ++ind2){
          vec[ind*row_X*batch_size+ind2] = W[i][j*n2+ind];
        }
      }
      
      PhantomPlaintext ecd_vec;
      encoder.encode(vec, 1, scale, ecd_vec);
      encryptor.encrypt(ecd_vec, output[i*num_ct_x+j]);
    }
  }

  return output;
}

inline vector<PhantomCiphertext> LoRA_B_packing_batch_complex(const vector<vector<double>> & W, 
  int batch_size, int row_X, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();
  int num_ct_x = row_X*num_col*batch_size/n;
  
  int num_ct = num_row*num_ct_x/2;

  vector<PhantomCiphertext> output(num_ct);
  for (int i = 0; i < num_row; ++i){
    int n2 = num_col/num_ct_x; 
    
    for (int j = 0 ; j < num_ct_x/2 ; ++j){
      vector<complex<double>> vec(n,0);
      for (int ind = 0 ; ind < n2 ; ++ind){
        for (int ind2 = 0 ; ind2 < (row_X*batch_size) ; ++ind2){
          vec[ind*row_X*batch_size+ind2].real(W[i][2*j*n2+ind]);
          vec[ind*row_X*batch_size+ind2].imag(W[i][(2*j+1)*n2+ind]);
        }
      }
      
      PhantomPlaintext ecd_vec;
      encoder.encode(vec, 1, scale, ecd_vec);
      encryptor.encrypt(ecd_vec, output[i*num_ct_x/2+j]);
    }
  }

  return output;
}

inline vector<PhantomCiphertext> Class_1_packing_batch(const vector<vector<double>> & W, 
  int batch_size, int row_X, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();
  int num_ct = num_row*num_col*batch_size/n;
  int n2 = n/(batch_size*row_X);

  vector<PhantomCiphertext> output(num_ct);
  for (int i = 0; i < num_col/row_X; ++i){
    for (int j = 0 ; j < num_row/n2 ; ++j){
      vector<double> vec(n,0);
      for (int ind1 = 0 ; ind1< n2 ; ++ind1){
        for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
          for (int ind3 = 0 ; ind3 < batch_size ; ++ind3){
            vec[ind1*row_X*batch_size+ind2*batch_size+ind3] = W[j*n2+ind1][i*row_X+ind2];
          }
          
        }
      }
      PhantomPlaintext ecd_vec;
      encoder.encode(vec, scale, ecd_vec);
      
      encryptor.encrypt(ecd_vec, output[i*num_row/n2+j]);
    }
  }

  return output;
}

inline PhantomCiphertext Class_2_packing_batch(const vector<vector<double>> & W, 
  int batch_size, int num_row, int num_col, int k,
  double scale, PhantomContext & seal_context,PhantomPublicKey & pk){
  PhantomCKKSEncoder phantom_encoder(seal_context);
    Encoder encoder(&seal_context, &phantom_encoder);
  Encryptor encryptor(&seal_context, &pk);
  int n = encoder.slot_count();

  PhantomCiphertext output;

  vector<double> vec(n,0);
  for (int i = 0; i < 2; ++i){
    for (int j = 0 ; j < num_row ; ++j){
      for (int k = 0 ; k < batch_size ; ++k){
        vec[i*n/2+j*batch_size+k] = W[j][i];
      }
    }
  }

  PhantomPlaintext ecd_vec;
  encoder.encode(vec, scale, ecd_vec);
  
  encryptor.encrypt(ecd_vec, output);

  return output;
}

inline vector<PhantomCiphertext> Complex_to_real(vector<PhantomCiphertext> & ct, 
  PhantomContext & seal_context, PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();
  int num_ct = ct.size();
  vector<PhantomCiphertext> output(2*num_ct);

  vector<double> r(n,0.5);
  PhantomPlaintext ecd_r;
  encoder.encode(r, ct[0].params_id(), ct[0].scale(), ecd_r);

  vector<complex<double>> r2(n, complex<double>{0,-0.5});
  
  PhantomPlaintext ecd_r2;
  encoder.encode(r2, ct[0].params_id(), ct[0].scale(), ecd_r2);

  for (int i = 0; i < num_ct; ++i){
    
    PhantomCiphertext tmp;
    evaluator.multiply_plain(ct[i],ecd_r,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    output[2*i] = tmp;
    evaluator.complex_conjugate_inplace(tmp,RotK);
    evaluator.add_inplace(output[2*i],tmp);

    evaluator.multiply_plain(ct[i],ecd_r2,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    output[2*i+1] = tmp;
    evaluator.complex_conjugate_inplace(tmp,RotK);
    evaluator.add_inplace(output[2*i+1],tmp);
  }

  return output;

}

inline vector<PhantomCiphertext> Complex_to_real_back(vector<PhantomCiphertext> & ct, double s,
  PhantomContext & seal_context, PhantomGaloisKey & RotK){

  PhantomCKKSEncoder phantom_encoder(seal_context);
  Encoder encoder(&seal_context, &phantom_encoder);
  Evaluator evaluator(&seal_context, &phantom_encoder);
  int n = encoder.slot_count();
  int num_ct = ct.size();
  vector<PhantomCiphertext> output(2*num_ct);

  vector<double> r(n,0.5/s);
  PhantomPlaintext ecd_r;
  encoder.encode(r, ct[0].params_id(), ct[0].scale(), ecd_r);

  vector<complex<double>> r2(n, complex<double>{0,-0.5/s});
  
  PhantomPlaintext ecd_r2;
  encoder.encode(r2, ct[0].params_id(), ct[0].scale(), ecd_r2);

  for (int i = 0; i < num_ct; ++i){
    
    PhantomCiphertext tmp;
    evaluator.multiply_plain(ct[i],ecd_r,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    output[2*i] = tmp;
    evaluator.complex_conjugate_inplace(tmp,RotK);
    evaluator.add_inplace(output[2*i],tmp);

    evaluator.multiply_plain(ct[i],ecd_r2,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    output[2*i+1] = tmp;
    evaluator.complex_conjugate_inplace(tmp,RotK);
    evaluator.add_inplace(output[2*i+1],tmp);
  }

  return output;

}

inline vector<vector<double>> Modified_col_packing_pt(const vector<vector<double>> & X, 
  int num_row, int num_col, int k, int n){

  int num_vec = num_col*num_row/n;

  vector<vector<double>> output(num_vec);

  for (int i = 0; i < num_vec; ++i){
    vector<double> vec(n,0);
    for (int j = 0 ; j < n ; ++j){
      int row_index = j % num_row;
      int col_index = k* ((j/num_row) % (n/(k*num_row))) + (k*j)/n + (i)*(n/num_row);
      vec[j] = X[row_index][col_index];
    }
    output[i] = vec;
  }
  
  return output;

}

inline vector<vector<double>> Modified_col_packing_batch_pt(const vector<vector<double>> & X, 
  int batch_size, int num_row, int num_col, int k, int n){

  int num_vec = num_col*num_row*batch_size/n;
  int n2 = n/batch_size;

  vector<vector<double>> output(num_vec);

  for (int i = 0; i < num_vec; ++i){
    vector<double> vec(n,0);
    for (int j = 0 ; j < n2 ; ++j){
      for (int ind = 0 ; ind < batch_size ; ++ind){
        int row_index = j % num_row;
        int col_index = k* ((j/num_row) % (n2/(k*num_row))) + (k*j)/n2 + (i)*(n2/num_row);
        vec[j*batch_size+ind] = X[row_index][col_index];
      }
    }
    output[i] = vec;
  }
  
  return output;

}
