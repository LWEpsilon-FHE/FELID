#pragma once
using namespace std;
using namespace seal;

inline vector<Ciphertext> gradient_LN1(const vector<Ciphertext> & gradient_att,  
  const vector<Ciphertext> & LN_no_norm, const Ciphertext & inv_sqrt_var, 
  const vector<double> & gamma, int batch_size, int num_col, int num_row, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  return gradient_LN(gradient_att, LN_no_norm, inv_sqrt_var, gamma, batch_size, num_col, num_row, seal_context, RotK, relin_keys);
}

inline vector<Ciphertext> gradient_cat(const vector<Ciphertext> & gradient_LN1,  
  const vector<vector<double>> & Wo, int batch_size, int num_col, int num_row, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<vector<double>> WoT = matrix_trans(Wo);
  
  vector<Ciphertext> LNWoT = PCMM_modified_col_packing_batch(gradient_LN1, WoT, batch_size, 
        768, 128, 768, 768, 1, seal_context,RotK);

  if(LNWoT.size()!=48){
    
  }

  return LNWoT;

}

inline vector<Ciphertext> gradient_U(const vector<Ciphertext> & gradient_cat, 
  const vector<Ciphertext> & V, int batch_size, int row, int col, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> rot_G=rot_G_bs(gradient_cat,batch_size,row,seal_context,RotK);

  return CCMM_GV_batch_bsgs(rot_G,V,batch_size,col,row,seal_context,RotK,relin_keys);

}

inline vector<Ciphertext> compute_D(const vector<Ciphertext> & gradient_U, const vector<Ciphertext> & ext_QK, 
  int batch_size, int row, const SEALContext& seal_context, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> output(row);
  for (int i = 0; i < row; ++i){
    evaluator.multiply(gradient_U[i], ext_QK[i], output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);
  }

  return output;

}

inline void align_gradient_operands(Ciphertext &a, Ciphertext &b, Evaluator &evaluator) {
  if(a.coeff_modulus_size()>b.coeff_modulus_size()) evaluator.mod_switch_to_inplace(a,b.parms_id());
  else if(b.coeff_modulus_size()>a.coeff_modulus_size()) evaluator.mod_switch_to_inplace(b,a.parms_id());
  a.scale()=b.scale();
}

inline vector<Ciphertext> gradient_Q(const vector<Ciphertext> & D, const vector<Ciphertext> & Q, const vector<Ciphertext> & rot_K,
  int batch_size, int row, int col, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> DK = CCMM_DK_batch_bsgs_complex(D,rot_K,batch_size,row,col,row,seal_context,RotK,relin_keys); 

  Ciphertext sum_D;
  for (int i = 0; i < row/16; ++i){
    Ciphertext tmp = D[i*16];
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

  vector<Ciphertext> real_Q = Complex_to_real_back(Q, 2.0, seal_context,RotK); 

  vector<Ciphertext> output(real_Q.size());
  for (int i = 0; i < real_Q.size(); ++i){
    Ciphertext lhs=sum_D,value=real_Q[i];
    align_gradient_operands(lhs,value,evaluator);
    evaluator.multiply(lhs,value,output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys); 
    
    evaluator.rescale_to_next_inplace(output[i]);
    Ciphertext product=DK[i];
    align_gradient_operands(output[i],product,evaluator);
    evaluator.sub_inplace(output[i],product);
  }

  return output;

}

inline vector<Ciphertext> gradient_K(const vector<Ciphertext> & D, const vector<Ciphertext> & K, const vector<Ciphertext> & Q,
  int batch_size, int row, int col, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> DT=rot_UT_gradV(D,batch_size,row,seal_context,RotK);

  vector<Ciphertext> DTQ = CCMM_V_batch(DT,Q,batch_size,
        row, col, row, 1, seal_context,RotK,relin_keys); 

  Ciphertext sum_DT;
  for (int i = 0; i < row/16; ++i){
    Ciphertext tmp = DT[i*16];
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

  vector<Ciphertext> real_K = Complex_to_real_back(K, 2.0, seal_context,RotK); 

  vector<Ciphertext> output(real_K.size());
  for (int i = 0; i < real_K.size(); ++i){
    Ciphertext lhs=sum_DT,value=real_K[i];
    align_gradient_operands(lhs,value,evaluator);
    evaluator.multiply(lhs,value,output[i]);
    
    evaluator.relinearize_inplace(output[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(output[i]);
    Ciphertext product=DTQ[i];
    align_gradient_operands(output[i],product,evaluator);
    evaluator.sub_inplace(output[i],product);
  }

  return output;

}

inline vector<Ciphertext> gradient_V(const vector<Ciphertext> & gradient_cat, 
  const vector<Ciphertext> & ext_U, int batch_size, int U_row, int num_col, int num_row,
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<Ciphertext> UT = rot_UT_gradV(ext_U,batch_size,U_row,seal_context,RotK);

  return CCMM_V_batch(UT, gradient_cat, batch_size,
        U_row, num_col, num_row, 1, seal_context,RotK,relin_keys);

}
