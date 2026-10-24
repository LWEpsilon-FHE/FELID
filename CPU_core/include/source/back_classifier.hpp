#pragma once
using namespace std;
using namespace seal;

inline Ciphertext partial_L_y(const Ciphertext & y_l, const Ciphertext & y_r, const SEALContext& seal_context){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  Ciphertext yl_minus_yr;
  evaluator.sub(y_l,y_r,yl_minus_yr);
  return yl_minus_yr;

}

inline Ciphertext back_class_2_batch(const Ciphertext & y_lr, const Ciphertext & tanhx, 
  int batch_size, int col_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  Ciphertext yl_minus_yr=y_lr;

  evaluator.multiply_inplace(yl_minus_yr,tanhx);
  evaluator.relinearize_inplace(yl_minus_yr,relin_keys);

  evaluator.rescale_to_next_inplace(yl_minus_yr);

  return yl_minus_yr;
}

inline vector<Ciphertext> pre_compute_back_class_1(const Ciphertext & y_lr, const Ciphertext & W2, 
  const Ciphertext & tanhx, 
  int batch_size, int col_W, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  Ciphertext tanh2;
  evaluator.square(tanhx,tanh2);
  
  evaluator.relinearize_inplace(tanh2, relin_keys);
  
  evaluator.rescale_to_next_inplace(tanh2);          

  evaluator.negate_inplace(tanh2);
  
  vector<double> one(n,1.0);
  Plaintext ecd1;
  encoder.encode(one, tanh2.parms_id(), tanh2.scale(), ecd1);
  evaluator.add_plain_inplace(tanh2,ecd1);

  Ciphertext output;

  evaluator.multiply(y_lr,W2,output);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);        

  Ciphertext tmp;
  evaluator.rotate_vector(output, n/2, RotK, tmp);
  evaluator.add_inplace(output,tmp);

  evaluator.multiply_inplace(output,tanh2);
  
  evaluator.relinearize_inplace(output,relin_keys);
  
  evaluator.rescale_to_next_inplace(output);

  vector<Ciphertext> final_output(6); 
  for (int i = 0; i < 6; ++i){
    vector<double> mask(n,0);
    for (int j = i*128*batch_size ; j < (i+1)*128*batch_size ; ++j){
      mask[j] = 1.0;
      mask[j+n/2] = 1.0;
    }
    Plaintext ecd_mask;
    encoder.encode(mask, output.parms_id(), output.scale(),ecd_mask);
    evaluator.multiply_plain(output, ecd_mask, final_output[i]);
    evaluator.rescale_to_next_inplace(final_output[i]);

    int rot_index = 128*batch_size;
    while(rot_index <= n/4){
      Ciphertext tmp2;
      evaluator.rotate_vector(final_output[i], rot_index, RotK, tmp2);
      evaluator.add_inplace(final_output[i], tmp2);
      rot_index *= 2;
    }
  }

  return final_output;

}

inline vector<Ciphertext> back_class_1_batch(const vector<Ciphertext> & ct, 
const vector<Ciphertext> & pre_compute_ct, 
  int batch_size, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> ext_ct = ext_copy_X(ct,batch_size, 768,128,seal_context,RotK);  

  vector<Ciphertext> output(ext_ct.size());
  for (int i = 0; i < 6; ++i){   
    for (int j = 0 ; j < 48 ; ++j){      
      evaluator.multiply(ext_ct[i*48+j], pre_compute_ct[i], output[i*48+j]);
      
      evaluator.relinearize_inplace(output[i*48+j],relin_keys);
      
      evaluator.rescale_to_next_inplace(output[i*48+j]);
    }
  }

  return output;

}

inline vector<Ciphertext> gradient_x2(const vector<Ciphertext> & W1, const vector<Ciphertext> & pre_compute_ct, 
  int batch_size, int col_W, int row_W, int k, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> output(48); 

  vector<double> mask(n,0.0);
  for (int i = 0; i < 16; ++i){
    for (int j = 0 ; j < batch_size ; ++j){
      mask[i*128*batch_size+j] = 1.0;
    }
  }
  Plaintext ecd_mask;
  encoder.encode(mask, pre_compute_ct[0].parms_id(), pre_compute_ct[0].scale(), ecd_mask);

  for (int i = 0; i < 48; ++i){
    
    for (int j = 0 ; j < 6 ; ++j){
      Ciphertext tmp;
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
      Ciphertext tmp;
      evaluator.rotate_vector(output[i], rot_index, RotK, tmp);
      
      evaluator.add_inplace(output[i],tmp);
      rot_index *= 2;
    }

    evaluator.mod_switch_to_inplace(ecd_mask,output[i].parms_id());
    evaluator.multiply_plain_inplace(output[i],ecd_mask);
    evaluator.rescale_to_next_inplace(output[i]);

  }

  return output;
}
