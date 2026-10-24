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

void Packing_test(){
    cout <<"Task: test modified column packing + encoding + encrypting: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

    struct timeval tstart1, tend1;

    //construct input
    int num_row = 128;
    int num_col = 768;
    int k = 1;

    vector<vector<double>> input_x(num_row, vector<double>(num_col,0));

    for (int i = 0 ; i < num_row ; ++i){
        for (int j = 0 ; j < num_col ; ++j){
            input_x[i][j] = (double)j+1.0+0.001*i;
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<endl;

    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<input_x[i][j]<<" ";
        }
        cout <<endl;
    }


    //encode + encrypt
    synchronize_test_device();
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> enc_ecd_x = Modified_col_packing(input_x, num_row, num_col, k, scale, context,public_key);

    synchronize_test_device();
    gettimeofday(&tend1,NULL);
    double enc_ecd_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"encode + encrypt time = "<<enc_ecd_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;


    cout <<"Decrypt + decode result: "<<endl;
    //decrypt
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    ErrorStats errors(1e-5, 0);
    for (int i = 0; i < enc_ecd_x.size(); ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        if (result.size() != slot_count) throw std::runtime_error("Unexpected slot count");
        for (std::size_t j = 0; j < slot_count; ++j) {
            const int column = i*(slot_count/num_row) + j/num_row;
            errors.add(result[j], input_x[j%num_row][column]);
        }
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 1024 ; ind += num_row){
            cout <<result[ind]<<" ... "<<result[ind+num_row-1]<<" ";
        }
        cout <<endl;
    }

    errors.report();
}

void Packing_batch_test(){
    cout <<"Task: test modified column packing + Interleaved batching + encoding + encrypting: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);

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
                input_x[ind][i][j] = (double)i+1.0+0.01*j+0.1*ind;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<endl;

    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<input_x[0][i][j]<<" ";
        }
        cout <<endl;
    }


    //encode + encrypt
    synchronize_test_device();
    gettimeofday(&tstart1,NULL);

    vector<PhantomCiphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    synchronize_test_device();
    gettimeofday(&tend1,NULL);
    double enc_ecd_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"encode + encrypt time = "<<enc_ecd_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(enc_ecd_x[0].params_id()).chain_depth()<<endl;


    cout <<"Decrypt + decode result: "<<endl;
    //decrypt
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    ErrorStats errors(1e-5, 0);
    for (int i = 0; i < enc_ecd_x.size(); ++i){
        PhantomPlaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        if (result.size() != slot_count) throw std::runtime_error("Unexpected slot count");
        for (std::size_t j = 0; j < slot_count; ++j) {
            const int column = i*(slot_count/(batch_size*num_row)) + j/(batch_size*num_row);
            errors.add(result[j], input_x[j%batch_size][(j/batch_size)%num_row][column]);
        }
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 33 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }

    errors.report();
}
