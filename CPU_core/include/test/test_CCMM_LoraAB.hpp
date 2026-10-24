#include "test_common.hpp"
using namespace std;
using namespace seal;



void CCMM_loraAB_test(){
    cout <<"Task: test Ct-Ct matrix multiplication for X * Lora A * Lora B: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    Evaluator evaluator(context, encoder);

    struct timeval tstart1, tend1;

//construct input
    int num_row = 128;
    int num_col = 768;
    int k = 1;
    int batch_size = 16;

    vector<vector<vector<double>>> input_x(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                input_x[ind][i][j] = (double)(i+j)*0.002+0.02;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<input_x[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;
   // cout <<"--------------------------------------"<<endl;



    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 2;

    vector<vector<double>> w(W_row, vector<double>(W_col,1.0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.001+i*0.001+j*0.002;
        }
    }

    cout <<"Matrix A size = "<<W_row <<" * "<<W_col<<endl;
/*
    cout <<"First 2 elements in the first 5 rows: "<<endl;
    for (int i = 0 ; i < 5 ; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout<<w[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_w = LoRA_A_packing_batch(w,batch_size,num_row,W_row,W_col,k,
        scale, context,public_key);

    cout <<"Modulus chain index for enc w: "<< context.get_context_data(enc_ecd_w[0].parms_id())->chain_index()<<endl;
   // cout <<"--------------------------------------"<<endl;

    vector<vector<double>> w2(W_col,vector<double>(W_row,1.0));

    for (int i = 0 ; i < W_col ; ++i){
        for (int j = 0 ; j < W_row ; ++j){
            w2[i][j] = 0.001+i*0.001+j*0.002;
        }
    }

    cout <<"Matrix B size = "<<W_col <<" * "<<W_row<<endl;

    vector<Ciphertext> enc_ecd_w2 = LoRA_B_packing_batch(w2,batch_size,num_row,W_col,W_row,k,
        scale, context,public_key);

    cout <<"Modulus chain index for enc w: "<< context.get_context_data(enc_ecd_w2[0].parms_id())->chain_index()<<endl;

  //  cout <<"--------------------------------------"<<endl;

/*
    for (int i = 0; i < enc_ecd_w.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_w[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res = MM(input_x[0],w);
/*
    cout <<"Real result of X * Lora A (10 * 2): "<<endl;
    for (int i = 0; i < 5; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }
    cout <<"..."<<endl;
    for (int i = num_row-5; i < num_row; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }

*/
    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(X)A..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_ct_mul = CCMM_LoRA_A_batch(enc_ecd_x, enc_ecd_w, batch_size,
        num_col, num_row, W_col, W_row, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! X*Lora A time = "<<ct_ct_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_ct_mul[0].parms_id())->chain_index()<<endl;

    vector<vector<double>> ct_res(ct_ct_mul.size());
//    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_ct_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_ct_mul[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 2 ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-2 ; ind < slot_count ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    cout <<"--------------------------------------"<<endl;

    vector<vector<double>> res2 = MM(res,w2);

    vector<vector<double>> res2_pack = Modified_col_packing_batch_pt(res2, batch_size, num_row, num_col, k, slot_count);

/*
    cout <<"Real result of X * Lora A * Lora B (10 * 10): "<<endl;
    for (int i = 0; i < 5; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res2[i][j]<<" ";
        }
        cout <<endl;
    }
    cout <<"..."<<endl;
    for (int i = num_row-5; i < num_row; ++i){
        for (int j = num_col-10 ; j < num_col ; ++j){
            cout <<res2[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    for (int i = 0; i < enc_ecd_w2.size(); ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_w2[i], ct_ct_mul[0].parms_id());
    }

    cout <<"Computing Enc(XA)*B..."<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_ct_mul2 = CCMM_LoRA_B_batch(ct_ct_mul,enc_ecd_w2,batch_size,
        W_col,num_row,W_row,W_col,k,context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! XA * Lora B time = "<<ct_ct_matrix_mul_time2<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_ct_mul2[0].parms_id())->chain_index()<<endl;

    vector<vector<double>> ct_res2(ct_ct_mul2.size());
//    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_ct_mul2.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_ct_mul2[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res2[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 2 ; ++ind){
            cout <<ct_res2[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-2 ; ind < slot_count ; ++ind){
            cout <<ct_res2[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    //compute error

    {
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res2.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res2[i][j], res2_pack[i][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;


}

void CCMM_loraAB_complex_test(){
    cout <<"Task: test Ct-Ct matrix multiplication for X * Lora A * Lora B(complex): "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    Evaluator evaluator(context, encoder);

    struct timeval tstart1, tend1;

//construct input
    int num_row = 128;
    int num_col = 768;
    int k = 1;
    int batch_size = 16;

    vector<vector<vector<double>>> input_x(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                input_x[ind][i][j] = (double)(i+j)*0.002+0.02;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<input_x[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;
   // cout <<"--------------------------------------"<<endl;



    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 2;

    vector<vector<double>> w(W_row, vector<double>(W_col,1.0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.001+i*0.001+j*0.002;
        }
    }

    cout <<"Matrix A size = "<<W_row <<" * "<<W_col<<endl;
/*
    cout <<"First 2 elements in the first 5 rows: "<<endl;
    for (int i = 0 ; i < 5 ; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout<<w[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_w = LoRA_A_packing_batch(w,batch_size,num_row,W_row,W_col,k,
        scale, context,public_key);

    cout <<"Modulus chain index for enc w: "<< context.get_context_data(enc_ecd_w[0].parms_id())->chain_index()<<endl;
   // cout <<"--------------------------------------"<<endl;

    vector<vector<double>> w2(W_col,vector<double>(W_row,1.0));

    for (int i = 0 ; i < W_col ; ++i){
        for (int j = 0 ; j < W_row ; ++j){
            w2[i][j] = 0.001+i*0.001+j*0.002;
        }
    }

    cout <<"Matrix B size = "<<W_col <<" * "<<W_row<<endl;

    vector<Ciphertext> enc_ecd_w2 = LoRA_B_packing_batch_complex(w2,batch_size,num_row,W_col,W_row,k,
        scale, context,public_key);

    cout <<"Modulus chain index for enc w: "<< context.get_context_data(enc_ecd_w2[0].parms_id())->chain_index()<<endl;

  //  cout <<"--------------------------------------"<<endl;

/*
    for (int i = 0; i < enc_ecd_w.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_w[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<"... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res = MM(input_x[0],w);
/*
    cout <<"Real result of X * Lora A (10 * 2): "<<endl;
    for (int i = 0; i < 5; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }
    cout <<"..."<<endl;
    for (int i = num_row-5; i < num_row; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }

*/
    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(X)A..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_ct_mul = CCMM_LoRA_A_batch(enc_ecd_x, enc_ecd_w, batch_size,
        num_col, num_row, W_col, W_row, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! X*Lora A time = "<<ct_ct_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_ct_mul[0].parms_id())->chain_index()<<endl;

    vector<vector<double>> ct_res(ct_ct_mul.size());
//    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_ct_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_ct_mul[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 2 ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-2 ; ind < slot_count ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    cout <<"--------------------------------------"<<endl;

    vector<vector<double>> res2 = MM(res,w2);

    vector<vector<double>> res2_pack = Modified_col_packing_batch_pt(res2, batch_size, num_row, num_col, k, slot_count);

/*
    cout <<"Real result of X * Lora A * Lora B (10 * 10): "<<endl;
    for (int i = 0; i < 5; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res2[i][j]<<" ";
        }
        cout <<endl;
    }
    cout <<"..."<<endl;
    for (int i = num_row-5; i < num_row; ++i){
        for (int j = num_col-10 ; j < num_col ; ++j){
            cout <<res2[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    for (int i = 0; i < enc_ecd_w2.size(); ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_w2[i], ct_ct_mul[0].parms_id());
    }

    cout <<"Computing Enc(XA)*B..."<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_ct_mul2 = CCMM_LoRA_B_batch_complex(ct_ct_mul,enc_ecd_w2,batch_size,
        W_col,num_row,W_row,W_col,k,context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! XA * Lora B time = "<<ct_ct_matrix_mul_time2<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_ct_mul2[0].parms_id())->chain_index()<<endl;

    vector<vector<complex<double>>> ct_res2(ct_ct_mul2.size());
//    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_ct_mul2.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_ct_mul2[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res2[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 2 ; ++ind){
            cout <<ct_res2[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-2 ; ind < slot_count ; ++ind){
            cout <<ct_res2[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    //compute error

    {
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res2.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res2[i][j].real(), res2_pack[2*i][j]);
            errors.add(ct_res2[i][j].imag(), res2_pack[2*i+1][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;


}
