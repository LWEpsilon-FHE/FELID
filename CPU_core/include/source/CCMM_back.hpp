#pragma once
using namespace std;
using namespace seal;

inline vector<Ciphertext> rot_G_bs(const vector<Ciphertext> & enc_G, 
  int batch_size, int row, const SEALContext& seal_context, 
  const GaloisKeys & RotK){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = enc_G.size();  
  int ed = row*row*batch_size/n;   
  int num_diag = row/ed;   

  vector<Ciphertext> rot_G(ed*mid);
  for(int ind = 0 ; ind < ed ; ++ind){
    for(int j = 0 ; j < mid ; ++j){
      rot_G[ind*mid+j] = enc_G[j];
      interal_rot_inplace(rot_G[ind*mid+j],(row-(ind*row/ed)),row,batch_size,1,seal_context,RotK); 
      
      pcm_num ++;
    }
  }

  return rot_G;

}

inline vector<Ciphertext> CCMM_GV_batch_bsgs(const vector<Ciphertext> & rot_G, const vector<Ciphertext> & enc_V, 
  int batch_size, int col, int row, 
  const SEALContext& seal_context, 
  const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  int pcm_num = 0;
  int ccm_num = 0;

  int mid = 4;  
  int ed = row*row*batch_size/n;   
  int num_diag = row/ed;   

  vector<Ciphertext> real_V = Complex_to_real_back(enc_V, (double)(-2*sqrt(128.0)), seal_context,RotK); 

  vector<Ciphertext> rot_V(num_diag*mid);
  for (int i = 0; i < num_diag; ++i){
    for (int j = 0 ; j < mid ; ++j){
      rot_V[i*mid+j] = real_V[j];
      interal_rot_inplace(rot_V[i*mid+j],i,row,batch_size,1,seal_context,RotK); 
      
      pcm_num ++;
    }
  }

  vector<Ciphertext> output(row);

  for (int ind = 0 ; ind < ed ; ++ind){
    for (int i = 0; i < num_diag; ++i){
      Ciphertext tmpoutput;
      for (int j = 0; j < mid; ++j){
        Ciphertext tmpGV;
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
          Ciphertext copy = tmpoutput;
          
          evaluator.rotate_vector_inplace(copy,sum_index,RotK);
          
          evaluator.add_inplace(tmpoutput,copy);
          sum_index /= 2;
      }
      output[ind*num_diag+i] = tmpoutput;
    }
  }

  return output;

}

inline vector<Ciphertext> CCMM_DK_batch_bsgs_complex(const vector<Ciphertext> & ext_D, const vector<Ciphertext> & rot_K, 
  int batch_size, int D_row, int K_col, int K_row,
  const SEALContext& seal_context, 
  const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  int ccm_num = 0;
  int pcm_num = 0;

  int midd = 8;
  int midk = 4;
  int ed = 4;
  int num_diag = 16;
  
  vector<Ciphertext> output(ed);

  for (int ind = 0; ind < midk; ++ind){

    for (int i = 0 ; i < midd ; ++i){
      Ciphertext tmpdk;
      for(int j = 0 ; j < num_diag ; ++j){
        Ciphertext tmp = ext_D[i*num_diag+j];

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

inline vector<Ciphertext> rot_UT_gradV( const vector<Ciphertext> & ext_U,
        int batch_size, int U_row, 
        const SEALContext& seal_context, const GaloisKeys & RotK){

        CKKSEncoder encoder(seal_context);
        Evaluator evaluator(seal_context, encoder);
        int n = encoder.slot_count();
        vector<Ciphertext> no_rot_UT(U_row);

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

inline vector<Ciphertext> CCMM_gradient_Lora(const vector<Ciphertext> & enc_X, const vector<Ciphertext> & pre_lora_gradients,
  int batch_size, int row, int col, int rank,
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys& relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  if(enc_X.size()!=48){
    
  }
  if(pre_lora_gradients.size() != rank){
    
  }

  int num_ct = enc_X.size();
  vector<Ciphertext> output(num_ct*rank);

  for (int i = 0; i < rank; ++i){
    for (int j = 0 ; j < num_ct ; ++j){
      
      evaluator.multiply(enc_X[j], pre_lora_gradients[i], output[i*num_ct+j]);
      
      evaluator.relinearize_inplace(output[i*num_ct+j],relin_keys);
      
      evaluator.rescale_to_next_inplace(output[i*num_ct+j]);
      
      int rot_index = row/2;
      while(rot_index >= 1){
        Ciphertext tmp = output[i*num_ct+j];
        interal_rot_inplace(tmp,rot_index,row,batch_size,1,seal_context,RotK);
        
        evaluator.add_inplace(output[i*num_ct+j],tmp);
        rot_index /= 2;
      }
    }
  }

  return output;

}
