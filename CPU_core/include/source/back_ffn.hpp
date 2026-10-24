#pragma once
using namespace std;
using namespace seal;

inline vector<vector<double>> matrix_trans(const vector<vector<double>> & input){
  vector<vector<double>> output(input[0].size(), vector<double>(input.size(),0));
  for (int i = 0; i < input[0].size(); ++i){
    for (int j = 0 ; j < input.size() ; ++j){
      output[i][j] = input[j][i];
    }
  }
  return output;
}

inline vector<Ciphertext> gradient_LN(const vector<Ciphertext> & gradient_x,
  const vector<Ciphertext> & normalized_x, const Ciphertext & inv_sqrt_var,
  const vector<double> & gamma, int batch_size, int num_col, int num_row,
  const SEALContext & seal_context, const GaloisKeys & RotK, const RelinKeys & relin_keys,
  bool input_is_normalized = true){
  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context,encoder);
  const int n=encoder.slot_count();
  const int count=gradient_x.size(),features=n/(batch_size*num_row);
  if(count<=0 || count!=normalized_x.size() || count*features!=num_col || gamma.size()!=num_col)
    throw std::invalid_argument("gradient_LN packing shape");
  auto align=[&](Ciphertext &a, Ciphertext &b){
    if(a.coeff_modulus_size()>b.coeff_modulus_size()) evaluator.mod_switch_to_inplace(a,b.parms_id());
    else if(b.coeff_modulus_size()>a.coeff_modulus_size()) evaluator.mod_switch_to_inplace(b,a.parms_id());
    a.scale()=b.scale();
  };
  auto reduce=[&](Ciphertext value){
    for(int step=n/2;step>=num_row*batch_size;step/=2){
      Ciphertext tmp;evaluator.rotate_vector(value,step,RotK,tmp);evaluator.add_inplace(value,tmp);
    }
    return value;
  };
  vector<Ciphertext> h(count);
  for(int i=0;i<count;++i){
    vector<double> slots(n);
    for(int j=0;j<n;++j) slots[j]=gamma[i*features+j/(num_row*batch_size)]/num_col;
    Plaintext coefficients;encoder.encode(slots,gradient_x[i].parms_id(),gradient_x[i].scale(),coefficients);
    evaluator.multiply_plain(gradient_x[i],coefficients,h[i]);evaluator.rescale_to_next_inplace(h[i]);
  }
  Ciphertext mean=h.front();
  for(int i=1;i<count;++i)evaluator.add_inplace(mean,h[i]);
  mean=reduce(mean);
  Ciphertext projection;
  for(int i=0;i<count;++i){
    Ciphertext z=normalized_x[i],g=h[i],product;align(z,g);evaluator.multiply(z,g,product);
    if(i==0)projection=product;else evaluator.add_inplace(projection,product);
  }
  evaluator.relinearize_inplace(projection,relin_keys);evaluator.rescale_to_next_inplace(projection);
  projection=reduce(projection);
  if(!input_is_normalized){
    Ciphertext inverse2=inv_sqrt_var;evaluator.square_inplace(inverse2);
    evaluator.relinearize_inplace(inverse2,relin_keys);evaluator.rescale_to_next_inplace(inverse2);
    align(projection,inverse2);evaluator.multiply_inplace(projection,inverse2);
    evaluator.relinearize_inplace(projection,relin_keys);evaluator.rescale_to_next_inplace(projection);
  }
  for(int i=0;i<count;++i){
    Ciphertext z=normalized_x[i],p=projection;align(z,p);
    evaluator.multiply_inplace(z,p);evaluator.relinearize_inplace(z,relin_keys);evaluator.rescale_to_next_inplace(z);
    Plaintext hidden_size;encoder.encode(vector<double>(n,num_col),h[i].parms_id(),h[i].scale(),hidden_size);
    evaluator.multiply_plain_inplace(h[i],hidden_size);evaluator.rescale_to_next_inplace(h[i]);
    Ciphertext m=mean;align(h[i],m);evaluator.sub_inplace(h[i],m);align(h[i],z);evaluator.sub_inplace(h[i],z);
    Ciphertext inv=inv_sqrt_var;align(h[i],inv);evaluator.multiply_inplace(h[i],inv);
    evaluator.relinearize_inplace(h[i],relin_keys);evaluator.rescale_to_next_inplace(h[i]);
  }
  return h;
}

inline vector<Ciphertext> gradient_att(const vector<Ciphertext> & gradient_LN, 
  const vector<Ciphertext> & tanh_gelu, const vector<Ciphertext> & gelu_input,  
  const vector<vector<double>> & wf1, const vector<vector<double>> & wf2,
  int batch_size, int num_col, int num_row, 
  const SEALContext& seal_context, const GaloisKeys & RotK, const RelinKeys &relin_keys){

  CKKSEncoder encoder(seal_context);
  Evaluator evaluator(seal_context, encoder);
  int n = encoder.slot_count();

  vector<vector<double>> wf2T = matrix_trans(wf2);
  
  vector<Ciphertext> Gzwf2T = PCMM_modified_col_packing_batch(gradient_LN, wf2T, batch_size,   
        768, 128, 3072, 768, 1, seal_context,RotK);

  for (int i = 0; i < 48*4; ++i){

    Ciphertext tanh2;
    evaluator.square(tanh_gelu[i],tanh2);
    
    evaluator.relinearize_inplace(tanh2, relin_keys);
    
    evaluator.rescale_to_next_inplace(tanh2);          

    evaluator.negate_inplace(tanh2);
    
    vector<double> one(n,1.0);
    Plaintext ecd1;
    encoder.encode(one, tanh2.parms_id(), tanh2.scale(), ecd1);
    evaluator.add_plain_inplace(tanh2,ecd1);

    Ciphertext x2;
    evaluator.square(gelu_input[i],x2);
    
    evaluator.relinearize_inplace(x2,relin_keys);  
    
    evaluator.rescale_to_next_inplace(x2);
    vector<double> c(n,0.134145*0.5*(sqrt(2.0/3.1415926)));
    encoder.encode(c,x2.parms_id(),x2.scale(),ecd1);
    evaluator.multiply_plain_inplace(x2,ecd1);
    evaluator.rescale_to_next_inplace(x2);     
    vector<double> c2(n,0.5*sqrt(2.0/3.1415926));
    encoder.encode(c2,x2.parms_id(),x2.scale(),ecd1);
    evaluator.add_plain_inplace(x2,ecd1);

    evaluator.mod_switch_to_inplace(tanh2,x2.parms_id());
    evaluator.multiply_inplace(tanh2,x2);
    
    evaluator.relinearize_inplace(tanh2,relin_keys);
    
    evaluator.rescale_to_next_inplace(tanh2);  

    vector<double> c3(n,0.5);
    encoder.encode(c3,tanh_gelu[i].parms_id(),tanh_gelu[i].scale(),ecd1);
    Ciphertext tmp;
    evaluator.multiply_plain(tanh_gelu[i],ecd1,tmp);
    evaluator.rescale_to_next_inplace(tmp);
    evaluator.add_plain_inplace(tmp,ecd1);

    evaluator.mod_switch_to_inplace(tmp,tanh2.parms_id());
    evaluator.add_inplace(tanh2,tmp);

    evaluator.multiply_inplace(Gzwf2T[i],tanh2);
    
    evaluator.relinearize_inplace(Gzwf2T[i],relin_keys);
    
    evaluator.rescale_to_next_inplace(Gzwf2T[i]);

  }

  vector<vector<double>> wf1T = matrix_trans(wf1);
  
  vector<Ciphertext> Gzwf1T = PCMM_modified_col_packing_batch(Gzwf2T, wf1T, batch_size,   
        3072, 128, 768, 3072, 1, seal_context,RotK);

  if(Gzwf1T.size() != 48){
    
  }

  for (int i = 0; i < 48; ++i){
    Ciphertext tmp = gradient_LN[i];
    evaluator.mod_switch_to_inplace(tmp, Gzwf1T[i].parms_id());
    evaluator.add_inplace(tmp,Gzwf1T[i]);
  }

  return Gzwf1T;

}
