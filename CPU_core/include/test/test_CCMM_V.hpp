#include "test_common.hpp"
using namespace std;
using namespace seal;

vector<vector<double>> kernel_diag_inv_pt(const vector<vector<double>> & Q, const vector<vector<double>> & K){
    int row = 128;
    int col = 64;
    vector<vector<double>> pt = kernel_pt(Q,K);
    vector<vector<double>> pt_diag(row, vector<double>(row,0.0));
    for (int i = 0; i < row; ++i){
        for (int j = 0 ; j < row ; ++j){
            pt_diag[j][i] = pt[j][(i+j)%row];
        }
    }
    return pt_diag;
}

void CCMM_Vtest(){
    cout <<"Task: test Ct-Ct matrix multiplication for Kernel * V: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 45, 45, 45, 45, 60}));
    double scale = pow(2.0,45);

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

    int num_row = 128;
    int num_col = 64;
    int k = 1;
    int batch_size = 16;

    vector<vector<double>> Q(num_row, vector<double>(num_col,0));

    for (int i = 0 ; i < num_row ; ++i){
        for (int j = 0 ; j < num_col ; ++j){
            Q[i][j] = (double)(i+j)*0.02+0.001;
        }
    }

   vector<vector<double>> K(num_row, vector<double>(num_col,0));

        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                K[i][j] = (double)(i)*0.02+0.01;
            }
        }


    vector<vector<double>> pt_result = kernel_diag_inv_pt(Q,K);
/*
    cout <<"QK: "<<endl;
    for (int i = 0; i < 10; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<pt_result[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<vector<double>>> input_x(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_row,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_row ; ++j){
                input_x[ind][i][j] = pt_result[i][j];
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;

    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_row, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    //construct V

    vector<vector<vector<double>>> V(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                V[ind][i][j] = (double)(i+j)*0.002+0.001;
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
    vector<Ciphertext> enc_ecd_V = Modified_col_packing_batch(V, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc V: "<< context.get_context_data(enc_ecd_V[0].parms_id())->chain_index()<<endl;



    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(||Q-K||)Enc(V)..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ext_QK = ext_QK_batch(enc_ecd_x,batch_size,num_row,context,gal_keys);

    vector<Ciphertext> output = CCMM_V_batch(ext_QK, enc_ecd_V, batch_size,
        num_row, num_col, num_row, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Enc(||Q-K||)*Enc(V) time = "<<ct_ct_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(output[0].parms_id())->chain_index()<<endl;


    //pt result
    vector<vector<double>> QK = kernel_pt(Q,K);
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
    vector<vector<double>> ct_res(output.size());
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
    for (int i = 0; i < ct_res.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res[i][j], res_pack[i][j]);
        }
    }
    errors.report();
    }




    cout <<"=========================TEST END======================="<<endl<<endl;





}
