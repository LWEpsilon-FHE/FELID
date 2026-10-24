#include "test_common.cuh"
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
#include "source/utils_fhe.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;
using namespace fhe;

vector<vector<double>> kernel_pt(const vector<vector<double>> & Q, const vector<vector<double>> & K){
    int row = 128;
    int col = 64;
    vector<vector<double>> output(row, vector<double>(row,0));
    for (int i = 0; i < row; ++i){
        for (int j = 0 ; j < row ; ++j){
            double sum = 0;
            for (int k = 0 ; k < col ; ++k){
                double tmp = Q[i][k]-K[j][k];
                tmp *= tmp;
                sum += tmp;
            }
            output[i][j] = sum;
        }
    }
    return output;
}

vector<vector<double>> kernel_diag_pt(const vector<vector<double>> & Q, const vector<vector<double>> & K){
    int row = 128;
    int pack_size = 16;
    int output_size = row/pack_size;
    int col = 64;
    vector<vector<double>> pt = kernel_pt(Q,K);
    vector<vector<double>> pt_diag(output_size, vector<double>(row*pack_size,0.0));
    for (int ind = 0 ; ind < output_size ; ++ind){
        for (int i = 0; i < pack_size; ++i){
            for (int j = 0 ; j < row ; ++j){
                pt_diag[ind][i*row+j] = pt[j][(ind*pack_size+i+j)%row];
            }
        }
    }
    return pt_diag;

}



void Internal_rot_test(){
    cout <<"Task: test Interal rotation: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);

    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();
    Evaluator evaluator(&context, &phantom_encoder);

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
                input_x[ind][i][j] = i+j+1+0.1*ind;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;

    vector<PhantomCiphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;
    cout <<"--------------------------------------"<<endl;

    cout <<"Decrypt + decode result: "<<endl;
    //decrypt

    PhantomPlaintext plain_result;
    decryptor.decrypt(enc_ecd_x[0], plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);

    for (int ind = 0 ; ind < num_row*batch_size ; ind += batch_size){
        cout <<result[ind]<<" ";
    }
    cout <<endl;

    interal_rot_inplace(enc_ecd_x[0],126,num_row,batch_size,k,context,gal_keys);


    cout <<"Modulus chain index for output: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;
    cout <<"--------------------------------------"<<endl;

    decryptor.decrypt(enc_ecd_x[0], plain_result);
    encoder.decode(plain_result, result);
    for (int ind = 0 ; ind < num_row*batch_size ; ind += batch_size){
        cout <<result[ind]<<" ";
    }
    cout <<endl;


    ErrorStats errors(1e-4, 0);
    if (result.size() != slot_count) throw std::runtime_error("Unexpected slot count");
    for (std::size_t s = 0; s < slot_count; ++s) {
        const int row = ((s/batch_size)%num_row + 126)%num_row;
        const int column = s/(batch_size*num_row);
        errors.add(result[s], input_x[s%batch_size][row][column]);
    }
    errors.report();
}

void CCMM_KernelQK_test(){
    cout <<"Task: test Ct-Ct matrix multiplication for ||Q-K||_2^2: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 40, 40, 60}));
    double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);

    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();
    Evaluator evaluator(&context, &phantom_encoder);

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
                Q[ind][i][j] = (double)(i+j)*0.02+0.001;
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
    vector<PhantomCiphertext> enc_ecd_Q = Modified_col_packing_batch(Q, batch_size,
     num_row, num_col, k, scale, context,public_key);

    for (int i = 0; i < enc_ecd_Q.size(); ++i){
        evaluator.mod_switch_to_next_inplace(enc_ecd_Q[i]);
    }

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_Q[0].params_id()).chain_depth()<<endl;
    //cout <<"--------------------------------------"<<endl;

    vector<vector<vector<double>>> K(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                K[ind][i][j] = (double)(i)*0.02+0.01;
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
    vector<PhantomCiphertext> enc_ecd_K = Modified_col_packing_batch(K, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_K[0].params_id()).chain_depth()<<endl;
    cout <<"--------------------------------------"<<endl;

    cout <<"Computing ||Q-K||_2^2..."<<endl;

    synchronize_test_device();
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> output = CCMM_Kernel_batch(enc_ecd_Q,enc_ecd_K,batch_size,num_col,num_row,k,context,gal_keys,relin_keys);

    synchronize_test_device();
    gettimeofday(&tend1,NULL);
    double QKkernel_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! QK kernel time = "<<QKkernel_time<<endl;
    //cout <<"Size of output = "<<output.size()<<endl;
    cout <<"Modulus chain index for output: "<< context.get_context_data(output[0].params_id()).chain_depth()<<endl;
    cout <<"--------------------------------------"<<endl;

 //   cout <<"Decrypt + decode result: "<<endl;
    vector<vector<double>> ct_res(output.size());
    //decrypt and decode
    for (int i = 0; i < output.size(); ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
/*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10*batch_size ; ind += batch_size){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
*/
    }

    vector<vector<double>> pt_result = kernel_diag_pt(Q[0],K[0]);
 //   cout <<"pt_result size = "<<pt_result.size()<<" * "<<pt_result[0].size()<<endl;
/*
    for (int i = 0; i < 10; ++i){
        for(int j = 0 ; j < 10; ++j){
            cout <<pt_result[i][j]<<" ";
        }
        cout <<endl;
*/
    //compute error

    {
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res.size(); ++i){
        for (int j = 0 ; j < num_row*16 ; ++j){
            for (int b = 0; b < batch_size; ++b)
                errors.add(ct_res[i][j*batch_size+b], pt_result[i][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;


}
