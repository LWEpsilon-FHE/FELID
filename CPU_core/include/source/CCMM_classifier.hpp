#pragma once
using namespace std;
using namespace seal;

inline vector<Ciphertext> ext_copy_X(const vector<Ciphertext> & enc_X, 
  int batch_size, int col_X, int row_X,
  const SEALContext& seal_context, const GaloisKeys & RotK){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<double> mask(n,0);
  for (int i = 0; i < n/(row_X*batch_size); ++i){
    for (int j = 0 ; j < batch_size ; ++j){
      mask[i*row_X*batch_size+j] = 1;
    }
  }

  Plaintext ecd_mask;
  encoder.encode(mask, enc_X[0].parms_id(), enc_X[0].scale(), ecd_mask);

  vector<Ciphertext> ext_X(enc_X.size());
  for (int i = 0; i < enc_X.size(); ++i){
    evaluator.multiply_plain(enc_X[i],ecd_mask,ext_X[i]);
    evaluator.rescale_to_next_inplace(ext_X[i]);             
    
    int rot_index = batch_size;
    while(rot_index < row_X*batch_size){
      Ciphertext tmp;
      evaluator.rotate_vector(ext_X[i],-rot_index,RotK,tmp);
      
      evaluator.add_inplace(ext_X[i],tmp);
      rot_index *= 2;
    }
  }

  return ext_X;

}

inline Ciphertext CCMM_class_1_batch(const vector<Ciphertext> & ext_X, 
  const vector<Ciphertext> & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, 
  const GaloisKeys & RotK, const RelinKeys &relin_keys){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  int mid = row_X*col_X*batch_size/n;
  int ed = W.size()/mid;

  vector<Ciphertext> output(ed);

  for (int i = 0; i < ed; ++i){
    
    for (int j = 0 ; j < mid ; ++j){
     
      Ciphertext tmp;
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
     
      Ciphertext tmp;
      evaluator.rotate_vector(output[i], rot_index, RotK, tmp);
      
      evaluator.add_inplace(output[i],tmp);
      rot_index /= 2;
    }
  }

  Ciphertext final_output;
  
  for (int i = 0; i < ed; ++i){
    
    vector<double> mask(n,0.0);
    for(int j = i*row_X*batch_size ; j < (i+1)*row_X*batch_size ; ++j){
      mask[j] = 1.0;
    }
    Plaintext ecd_mask;
    encoder.encode(mask, output[i].parms_id(), output[i].scale(), ecd_mask);
    evaluator.multiply_plain_inplace(output[i],ecd_mask);
    evaluator.rescale_to_next_inplace(output[i]);  

    if(i == 0){
      final_output = output[i];
    }
    else{
      evaluator.add_inplace(final_output,output[i]);
    }
  }

  Ciphertext tmp;
  evaluator.rotate_vector(final_output,n/2,RotK,tmp);
  
  evaluator.add_inplace(final_output,tmp);

  return final_output;

}

inline Ciphertext CCMM_class_2_batch(const Ciphertext & enc_X, 
  const Ciphertext & W, int batch_size,
  int col_X, int row_X, int col_W, int row_W, int k, 
  const SEALContext& seal_context, 
  const GaloisKeys & RotK, const RelinKeys &relin_keys){

  if(col_X != row_W){
    
  }

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  Ciphertext output = enc_X;
  
  evaluator.multiply_inplace(output, W);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);

  int rot_index = n/4;
  while(rot_index >= batch_size){
    Ciphertext tmp;
    evaluator.rotate_vector(output,rot_index,RotK,tmp);
    
    evaluator.add_inplace(output,tmp);
    rot_index /= 2;
  }

  return output;
}
