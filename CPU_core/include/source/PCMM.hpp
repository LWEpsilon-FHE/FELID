#pragma once
using namespace std;
using namespace seal;

inline vector<double> Vec_left_rot(const vector<double> v, int r){
  int size = v.size();
  vector<double> output(size,0);

  for (int i = 0; i < size; ++i){
    output[i] = v[((i+r) % size + size) % size ];
  }

  return output;
}

inline vector<complex<double>> Vec_left_rot_complex(const vector<complex<double>> v, int r){
  int size = v.size();
  vector<complex<double>> output(size,0);

  for (int i = 0; i < size; ++i){
    output[i] = v[((i+r) % size + size) % size ];
  }

  return output;
}

inline vector<Ciphertext> PCMM_modified_col_packing(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, 
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  int mid = row_X*col_X/n;
  int ed = row_X*col_W/n;
  
  vector<Ciphertext> output(ed);

  int N1 = 0;
  int N2 = 0;
  if(row_X == 128){ 
    N1 = 16;
    N2 = 16;
  }

  vector<Ciphertext> ct_N1(mid*(N1));

  #pragma omp parallel for
  for (int j = 0 ; j < mid ; ++j){
    for (int q = 0 ; q < N1  ; ++q){
      evaluator.rotate_vector(enc_X[j],q*row_X,RotK,ct_N1[j*(N1)+q]);
    }
  }

  for (int l = 0 ; l < ed ; ++l){
    for (int p = 0 ; p < N2 ; ++p){
      Ciphertext tmp;
      for (int j = 0 ; j < mid ; ++j){
        for(int q = 0 ; q < N1 ; ++q){
          
          vector<double> d(n,0);
          int i = p*N1+q;
          for (int ind1 = 0 ; ind1 < (n/row_X) ; ++ind1){
            for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
              d[ind1*row_X+ind2] = W[j*n/row_X + (i+ind1)%(n/row_X)][l*n/row_X + ind1%(n/row_X)];
            }
          }

          Plaintext pt_d;
          encoder.encode(Vec_left_rot(d,-1*p*N1*row_X), ct_N1[j*(N1)+q].parms_id(), ct_N1[j*(N1)+q].scale(), pt_d);
          Ciphertext ct_d;
          evaluator.multiply_plain(ct_N1[j*(N1)+q],pt_d,ct_d);

          if(j ==0 && q == 0){
            tmp = ct_d;
          }
          else{
            evaluator.add_inplace(tmp,ct_d);
          }
        }
      }
      
      evaluator.rotate_vector_inplace(tmp,p*N1*row_X,RotK);
      if(p == 0){
        output[l] = tmp;
      }
      else{
        evaluator.add_inplace(output[l],tmp);
      }
    }
    evaluator.rescale_to_next_inplace(output[l]);
  }

  return output;

}

inline vector<Ciphertext> PCMM_modified_col_packing_batch(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();
  int n2 = n/batch_size;

  int mid = row_X*col_X*batch_size/n; 
  int ed = row_X*col_W*batch_size/n;
  
  vector<Ciphertext> output(ed);

  int mul_num = 0;

  int N1 = 0;
  int N2 = 0;
  if(row_X == 128 && batch_size == 16){ 
    N1 = 4;
    N2 = 4;
  }

  vector<Ciphertext> ct_N1(mid*(N1));

  #pragma omp parallel for
  for (int j = 0 ; j < mid ; ++j){
    for (int q = 0 ; q < N1  ; ++q){
      evaluator.rotate_vector(enc_X[j],q*row_X*batch_size,RotK,ct_N1[j*(N1)+q]);
      
    }
  }

  for (int l = 0 ; l < ed ; ++l){
    
    for (int p = 0 ; p < N2 ; ++p){
      Ciphertext tmp;
      for (int j = 0 ; j < mid ; ++j){
        for(int q = 0 ; q < N1 ; ++q){
          
          vector<double> d(n,0);
          int i = p*N1+q;
          for (int ind1 = 0 ; ind1 < (n2/row_X) ; ++ind1){
            for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
              for (int ind3 = 0 ; ind3 < batch_size ; ++ind3){
                d[ind1*row_X*batch_size+ind2*batch_size+ind3] 
                  = W[j*n2/row_X + (i+ind1)%(n2/row_X)][l*n2/row_X + ind1%(n2/row_X)];
              }
            }
          }

          Plaintext pt_d;
          encoder.encode(Vec_left_rot(d,-1*p*N1*row_X*batch_size), ct_N1[j*(N1)+q].parms_id(), ct_N1[j*(N1)+q].scale(), pt_d);
          Ciphertext ct_d;
          evaluator.multiply_plain(ct_N1[j*(N1)+q],pt_d,ct_d);
          mul_num++;

          if(j ==0 && q == 0){
            tmp = ct_d;
          }
          else{
            evaluator.add_inplace(tmp,ct_d);
          }
        }
      }
      
      evaluator.rotate_vector_inplace(tmp,p*N1*row_X*batch_size,RotK);
      
      if(p == 0){
        output[l] = tmp;
      }
      else{
        evaluator.add_inplace(output[l],tmp);
      }
    }
    evaluator.rescale_to_next_inplace(output[l]);   
  }

  return output;

}

inline vector<Ciphertext> PCMM_modified_col_packing_batch_complex(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();
  int n2 = n/batch_size;

  int mid = row_X*col_X*batch_size/n; 
  int ed = row_X*col_W*batch_size/(2*n);
  
  vector<Ciphertext> output(ed);

  int mul_num = 0;

  int N1 = 0;
  int N2 = 0;
  if(row_X == 128 && batch_size == 16){ 
    N1 = 4;
    N2 = 4;
  }

  vector<Ciphertext> ct_N1(mid*(N1));

  #pragma omp parallel for
  for (int j = 0 ; j < mid ; ++j){
    for (int q = 0 ; q < N1  ; ++q){
      evaluator.rotate_vector(enc_X[j],q*row_X*batch_size,RotK,ct_N1[j*(N1)+q]);
      
    }
  }

  for (int l = 0 ; l < ed ; ++l){
    
    for (int p = 0 ; p < N2 ; ++p){
      Ciphertext tmp;
      for (int j = 0 ; j < mid ; ++j){
        for(int q = 0 ; q < N1 ; ++q){
          
          vector<complex<double>> d(n,0);
          int i = p*N1+q;
          for (int ind1 = 0 ; ind1 < (n2/row_X) ; ++ind1){
            for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
              for (int ind3 = 0 ; ind3 < batch_size ; ++ind3){
                d[ind1*row_X*batch_size+ind2*batch_size+ind3].real( 
                  W[j*n2/row_X + (i+ind1)%(n2/row_X)][2*l*n2/row_X + ind1%(n2/row_X)]);
                d[ind1*row_X*batch_size+ind2*batch_size+ind3].imag( 
                  W[j*n2/row_X + (i+ind1)%(n2/row_X)][(2*l+1)*n2/row_X + ind1%(n2/row_X)]);
              }
            }
          }

          Plaintext pt_d;
          encoder.encode(Vec_left_rot_complex(d,-1*p*N1*row_X*batch_size), ct_N1[j*(N1)+q].parms_id(), ct_N1[j*(N1)+q].scale(), pt_d);
          Ciphertext ct_d;
          evaluator.multiply_plain(ct_N1[j*(N1)+q],pt_d,ct_d);
          mul_num++;

          if(j ==0 && q == 0){
            tmp = ct_d;
          }
          else{
            evaluator.add_inplace(tmp,ct_d);
          }
        }
      }
      
      evaluator.rotate_vector_inplace(tmp,p*N1*row_X*batch_size,RotK);
      
      if(p == 0){
        output[l] = tmp;
      }
      else{
        evaluator.add_inplace(output[l],tmp);
      }
    }
    evaluator.rescale_to_next_inplace(output[l]);   
  }

  return output;

}

inline vector<Ciphertext> PCMM_selfoutput_batch_complex(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();
  int n2 = n/batch_size;

  int mid = row_X*col_X*batch_size/n; 
  int ed = row_X*col_W*batch_size/(2*n);
  
  vector<Ciphertext> output(ed);

  int mul_num = 0;

  int N1 = 0;
  int N2 = 0;
  if(row_X == 128 && batch_size == 16){ 
    N1 = 4;
    N2 = 4;
  }

  vector<Ciphertext> ct_N1(mid*(N1));

  for (int j = 0 ; j < mid ; ++j){
    for (int q = 0 ; q < N1  ; ++q){
      evaluator.rotate_vector(enc_X[j],q*row_X*batch_size,RotK,ct_N1[j*(N1)+q]);
      
    }
  }

  for (int l = 0 ; l < ed ; ++l){
    
    for (int p = 0 ; p < N2 ; ++p){
      Ciphertext tmp;
      for (int j = 0 ; j < mid ; ++j){
        for(int q = 0 ; q < N1 ; ++q){
          
          vector<complex<double>> d(n,0);
          int i = p*N1+q;
          for (int ind1 = 0 ; ind1 < (n2/row_X) ; ++ind1){
            for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
              for (int ind3 = 0 ; ind3 < batch_size ; ++ind3){
                if((j)%2 == 0){      
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].real( 
                    0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][2*l*n2/row_X + ind1%(n2/row_X)]);
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].imag( 
                    0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][(2*l+1)*n2/row_X + ind1%(n2/row_X)]);
                }
                else{
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].imag( 
                    -0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][2*l*n2/row_X + ind1%(n2/row_X)]);
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].real( 
                    0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][(2*l+1)*n2/row_X + ind1%(n2/row_X)]);
                }
                
              }
            }
          }

          Plaintext pt_d;
          encoder.encode(Vec_left_rot_complex(d,-1*p*N1*row_X*batch_size), ct_N1[j*(N1)+q].parms_id(), ct_N1[j*(N1)+q].scale(), pt_d);
          Ciphertext ct_d;
          evaluator.multiply_plain(ct_N1[j*(N1)+q],pt_d,ct_d);
          mul_num++;

          if(j ==0 && q == 0){
            tmp = ct_d;
          }
          else{
            evaluator.add_inplace(tmp,ct_d);
          }
        }
      }
      
      evaluator.rotate_vector_inplace(tmp,p*N1*row_X*batch_size,RotK);
      
      if(p == 0){
        output[l] = tmp;
      }
      else{
        evaluator.add_inplace(output[l],tmp);
      }
    }
    evaluator.rescale_to_next_inplace(output[l]);   
  }

  return output;

}

inline vector<Ciphertext> PCMM_selfoutput_batch(const vector<Ciphertext> & enc_X, 
  const vector<vector<double>> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();
  int n2 = n/batch_size;

  int mid = row_X*col_X*batch_size/n; 
  int ed = row_X*col_W*batch_size/(n);
  
  vector<Ciphertext> output(ed);

  int mul_num = 0;

  int N1 = 0;
  int N2 = 0;
  if(row_X == 128 && batch_size == 16){ 
    N1 = 4;
    N2 = 4;
  }

  vector<Ciphertext> ct_N1(mid*(N1));

  for (int j = 0 ; j < mid ; ++j){
    for (int q = 0 ; q < N1  ; ++q){
      evaluator.rotate_vector(enc_X[j],q*row_X*batch_size,RotK,ct_N1[j*(N1)+q]);
      
    }
  }

  for (int l = 0 ; l < ed ; ++l){
    
    for (int p = 0 ; p < N2 ; ++p){
      Ciphertext tmp;
      for (int j = 0 ; j < mid ; ++j){
        for(int q = 0 ; q < N1 ; ++q){
          
          vector<complex<double>> d(n,0);
          int i = p*N1+q;
          for (int ind1 = 0 ; ind1 < (n2/row_X) ; ++ind1){
            for (int ind2 = 0 ; ind2 < row_X ; ++ind2){
              for (int ind3 = 0 ; ind3 < batch_size ; ++ind3){
                if((j)%2 == 0){      
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].real( 
                    0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][l*n2/row_X + ind1%(n2/row_X)]);
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].imag(0);
                }
                else{
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].imag( 
                    -0.5*W[j*n2/row_X + (i+ind1)%(n2/row_X)][l*n2/row_X + ind1%(n2/row_X)]);
                  d[ind1*row_X*batch_size+ind2*batch_size+ind3].real(0);
                }
                
              }
            }
          }

          Plaintext pt_d;
          encoder.encode(Vec_left_rot_complex(d,-1*p*N1*row_X*batch_size), ct_N1[j*(N1)+q].parms_id(), ct_N1[j*(N1)+q].scale(), pt_d);
          Ciphertext ct_d;
          evaluator.multiply_plain(ct_N1[j*(N1)+q],pt_d,ct_d);
          mul_num++;

          if(j ==0 && q == 0){
            tmp = ct_d;
          }
          else{
            evaluator.add_inplace(tmp,ct_d);
          }
        }
      }
      
      evaluator.rotate_vector_inplace(tmp,p*N1*row_X*batch_size,RotK);
      
      if(p == 0){
        output[l] = tmp;
      }
      else{
        evaluator.add_inplace(output[l],tmp);
      }
    }
    evaluator.rescale_to_next_inplace(output[l]);   
  }

  return output;

}
