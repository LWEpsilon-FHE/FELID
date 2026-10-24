#include "test_common.hpp"
using namespace std;
using namespace seal;

void KernelQK_V_bsgs_selfoutput_test(){
    cout <<"Task: test ||Q-K||_2^2*V*W_o: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
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
    int num_col = 64;
    int k = 1;
    int batch_size = 16;

    vector<vector<vector<double>>> Q(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                Q[ind][i][j] = (double)(i+j)*0.0002+0.0001;
            }
        }
    }

    cout <<"Matrix Q size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<Q[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_Q = Modified_col_packing_batch(Q, batch_size,
     num_row, num_col, k, scale, context,public_key);

//    for (int i = 0; i < enc_ecd_Q.size(); ++i){
 //       evaluator.mod_switch_to_next_inplace(enc_ecd_Q[i]);
 //   }

    cout <<"Modulus chain index for enc Q: "<< context.get_context_data(enc_ecd_Q[0].parms_id())->chain_index()<<endl;
    //cout <<"--------------------------------------"<<endl;

    vector<vector<vector<double>>> K(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                K[ind][i][j] = (double)(i)*0.0002+0.0001;
            }
        }
    }
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<K[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_K = Modified_col_packing_batch(K, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc K: "<< context.get_context_data(enc_ecd_K[0].parms_id())->chain_index()<<endl;
    cout <<"--------------------------------------"<<endl;

    cout <<"Computing ||Q-K||_2^2..."<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> encQK = CCMM_Kernel_batch_bsgs(enc_ecd_Q,enc_ecd_K,batch_size,num_col,num_row,k,context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double QKkernel_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! QK kernel time = "<<QKkernel_time<<endl;
    //cout <<"Size of output = "<<output.size()<<endl;
    cout <<"Modulus chain index for output: "<< context.get_context_data(encQK[0].parms_id())->chain_index()<<endl;
    cout <<"--------------------------------------"<<endl;

     //   cout <<"Decrypt + decode result: "<<endl;
    vector<vector<double>> ct_res_qk(encQK.size());
    //decrypt and decode
    for (int i = 0; i < encQK.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(encQK[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res_qk[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 130*batch_size ; ind += batch_size){
            cout <<ct_res_qk[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    //ADD functional bootstrapping here

    //construct V

    vector<vector<vector<double>>> V(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                V[ind][i][j] = (double)(i+j)*0.0002+0.0001;
            }
        }
    }

    cout <<"Matrix V size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<Q[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<Ciphertext> enc_ecd_V = Modified_col_packing_batch_complex(V, batch_size,
     num_row, num_col, k, scale, context,public_key);

    for (int i = 0; i < enc_ecd_V.size(); ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_V[i],encQK[0].parms_id());
    }

    cout <<"Modulus chain index for enc V: "<< context.get_context_data(enc_ecd_V[0].parms_id())->chain_index()<<endl;

     cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(||Q-K||)Enc(V)..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ext_QK = ext_QK_batch(encQK,batch_size,num_row,context,gal_keys);

    vector<Ciphertext> output = CCMM_V_batch_bsgs_complex(ext_QK, enc_ecd_V, batch_size,
        num_row, num_col, num_row, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Enc(||Q-K||)*Enc(V) time = "<<ct_ct_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(output[0].parms_id())->chain_index()<<endl;


    //pt result
    vector<vector<double>> QK = kernel_pt(Q[0],K[0]);
    vector<vector<double>> QKV= MM(QK,V[0]);
/*
    cout <<"QKV: First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<QKV[i][j]<<" ";
        }
        cout <<endl;
    }
    cout <<endl;
*/
    vector<vector<double>> res_pack = Modified_col_packing_batch_pt(QKV,batch_size,num_row, num_col, k, slot_count);
/*
    cout <<"packed pt result: "<<endl;
    for (int i = 0; i < res_pack.size(); ++i){
        for (int ind = 128*batch_size ; ind < 138*batch_size ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<endl;
    }
*/
 //   cout <<"Decrypt + decode result: "<<endl;
    vector<vector<complex<double>>> ct_res(output.size());
    //decrypt and decode
    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 128*batch_size ; ind < 138*batch_size ; ind += batch_size){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<endl;
*/
    }

    //compute error

    {
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res.size()/2; ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res[2*i][j].real()/2, res_pack[2*i][j]);
            errors.add(ct_res[2*i+1][j].imag()/2, res_pack[2*i+1][j]);
        }
    }
    errors.report();
    }

    //*selfoutput
    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 768;

    vector<vector<double>> w(W_row, vector<double>(W_col,0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.0001+j*0.0001+i*0.0002;
        }
    }

    cout <<"Matrix W_o size = "<<W_row <<" * "<<W_col<<endl;

    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(||Q-K||)Enc(V)*W_o..."<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_pt_mul = PCMM_selfoutput_batch(output, w,
        batch_size, num_col, num_row, W_col, W_row, k, context,gal_keys);

    gettimeofday(&tend1,NULL);
    double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_pt_mul[0].parms_id())->chain_index()<<endl;

     vector<vector<double>> res_wo = MM(QKV,w);
/*
     cout <<"Real result (10 * 10): "<<endl;
    for (int i = 0; i < 10; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res_wo[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res_pack_wo = Modified_col_packing_batch_pt(res_wo, batch_size, num_row, W_col, k, slot_count);
/*
    for (int i = 0; i < res_pack_wo.size(); ++i){
        cout <<i<<"-th vector: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<res_pack_wo[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<res_pack_wo[i][ind*batch_size]<<" ";
        }
        cout <<endl;
    }
*/
    /*
    vector<vector<complex<double>>> ct_res_wo(ct_pt_mul.size());
 //   cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_pt_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res_wo[i]);

        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<ct_res_wo[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<ct_res_wo[i][ind*batch_size]<<" ";
        }
        cout <<endl;
        }
 */

/*
    //compute error
    ave_error = 0.0;
    max_error = 0.0;

    for (int i = 0; i < ct_res_wo.size(); ++i){
        for (int j = 0 ; j < slot_count ; j += batch_size){
            double err = ct_res_wo[i][j].real() - res_pack_wo[2*i][j];
            ave_error += err;
            if(err > max_error){
                max_error = err;
            }
            err = ct_res_wo[i][j].imag() - res_pack_wo[2*i+1][j];
            ave_error += err;
            if(err > max_error){
                max_error = err;
            }
        }
    }

    ave_error /= (double)(2*ct_res_wo.size()*slot_count/batch_size);

    cout <<"average error = "<<ave_error<<endl;
    cout <<"max error = "<<max_error<<endl;

    var_error = 0.0;
    for (int i = 0; i < ct_res_wo.size(); ++i){
        for (int j = 0 ; j < slot_count ; j += batch_size){
            double var_err = (ct_res_wo[i][j].real() - res_pack_wo[2*i][j])-ave_error;
            var_error += (var_err*var_err);
            var_error = (ct_res_wo[i][j].imag() - res_pack_wo[2*i+1][j])-ave_error;
            var_error += (var_err*var_err);
        }
    }
    var_error /= (double)(2*ct_res_wo.size()*slot_count/batch_size);

    cout <<"variance of errors = "<<var_error<<endl;

    //complex_to_real(XW_o)
    cout <<"--------------------------------------"<<endl;
    cout<<"Computing converting complex packing to real parts..."<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> real_output = Complex_to_real(ct_pt_mul,context,gal_keys);

    gettimeofday(&tend1,NULL);
    double convert_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Ct-Pt matrix multiplication time = "<<convert_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(real_output[0].parms_id())->chain_index()<<endl;
*/

        vector<vector<double>> ct_res_wo_real(ct_pt_mul.size());
 //   cout <<"Decrypt + decode result of Complex to real: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_pt_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res_wo_real[i]);
 /*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<ct_res_wo_real[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<ct_res_wo_real[i][ind*batch_size]<<" ";
        }
        cout <<endl;
 */
    }


    {
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res_wo_real.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res_wo_real[i][j], res_pack_wo[i][j]);
        }
    }
    errors.report();
    }






    cout <<"=========================TEST END======================="<<endl<<endl;


}
