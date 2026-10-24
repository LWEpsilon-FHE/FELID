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

void Softmax_test(){
    cout <<"Task: test Softmax in classification head: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,60}));
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
    Encryptor encryptor(&context, &public_key);
    size_t slot_count = encoder.slot_count();
    Evaluator evaluator(&context, &phantom_encoder);

    struct timeval tstart1, tend1;

    //construct input
    vector<double> input(2,0);
    input[0] = 0.9;
    input[1] = 0.2;
    double max = 0.9;


    vector<double> output_pt = softmax_pt(input);
    cout <<"plaintext result = "<<output_pt[0]<<" "<<output_pt[1]<<endl;

    vector<double> x(slot_count,0);
    for (int i = 0; i < slot_count/2; ++i){
        x[i] = input[0];
        x[i+slot_count/2] = input[1];
    }
    PhantomPlaintext ecd_x;
    encoder.encode(x,scale,ecd_x);
    PhantomCiphertext enc_ecd_x;
    encryptor.encrypt(ecd_x, enc_ecd_x);

    cout <<"Modulus chain index for the enc x: "<< context.get_context_data(enc_ecd_x.params_id()).chain_depth()<<endl;

    synchronize_test_device();
    gettimeofday(&tstart1,NULL);

    PhantomCiphertext output = softmax(enc_ecd_x,context,relin_keys,gal_keys,10,max);

    synchronize_test_device();
    gettimeofday(&tend1,NULL);
    cout <<"Modulus chain index for the output: "<< context.get_context_data(output.params_id()).chain_depth()<<endl;
    double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Softmax time = "<<softmax_time<<endl;

    cout <<"Decrypt + decode result: "<<endl;
    PhantomPlaintext plain_result;
    decryptor.decrypt(output, plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);
    cout <<result[0]<<" "<<result[slot_count/2]<<endl;
    ErrorStats errors(1e-3, 0);
    if (result.size() != slot_count) throw std::runtime_error("Unexpected slot count");
    for (std::size_t j = 0; j < slot_count; ++j)
        errors.add(result[j], output_pt[j/(slot_count/2)]);
    errors.report();


 }

 void Softmax_boot_test(){
    cout <<"Task: test Softmax_boot in classification head: "<<endl;

      //bootstrapping parameters
    long boundary_K = 25;
  long deg = 59;
  long scale_factor = 2;
  long inverse_deg = 1;

  long logN = 16;
  long loge = 10;

  long logn = 15;
  long sparse_slots = (1 << logn);

  int logp = 46;
  int logq = 51;
  int log_special_prime = 58;

  int secret_key_hamming_weight = 192;

  // Calculation required
  int remaining_level = 12;
  int boot_level = 14;  // >= subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
  int total_level = remaining_level + boot_level;

  vector<int> coeff_bit_vec;
  coeff_bit_vec.push_back(logq);
  for (int i = 0; i < remaining_level; i++) {
    coeff_bit_vec.push_back(logp);
  }
  for (int i = 0; i < boot_level; i++) {
    coeff_bit_vec.push_back(logq);
  }
  coeff_bit_vec.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = (size_t)(1 << logN);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    double scale = pow(2.0, logp);

    PhantomContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);

    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();
    cout <<"slot count = "<<slot_count<<endl;

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &phantom_encoder, &relin_keys, &gal_keys, scale);
    Bootstrapper bootstrapper(
      loge,
      logn,
      logN - 1,
      total_level,
      scale,
      boundary_K,
      deg,
      scale_factor,
      inverse_deg,
      &ckks_evaluator);

  cout << "Generating Optimal Minimax Polynomials..." << endl;
  bootstrapper.prepare_mod_polynomial();

  cout << "Adding Bootstrapping Keys..." << endl;
  vector<int> gal_steps_vector;
  gal_steps_vector.push_back(0);
  for (int i = 0; i < logN - 1; i++) {
    gal_steps_vector.push_back((1 << i));
  }
  bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

  gal_keys = secret_key.create_galois_keys_from_steps(context, gal_steps_vector);

  bootstrapper.slot_vec.push_back(logn);

  cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();


    struct timeval tstart1, tend1;

    //construct input
    vector<double> input(2,0);
    input[0] = 0.9;
    input[1] = 0.2;
    double max = 0.9;


    vector<double> output_pt = softmax_pt(input);
    cout <<"plaintext result = "<<output_pt[0]<<" "<<output_pt[1]<<endl;

    vector<double> x(slot_count,0);
    for (int i = 0; i < slot_count/2; ++i){
        x[i] = input[0];
        x[i+slot_count/2] = input[1];
    }
    PhantomPlaintext ecd_x;
    encoder.encode(x,scale,ecd_x);
    PhantomCiphertext enc_ecd_x;
    encryptor.encrypt(ecd_x, enc_ecd_x);

    for (int i = 0; i < boot_level+3; ++i)
    {
        evaluator.mod_switch_to_next_inplace(enc_ecd_x);
    }

    cout <<"Modulus chain index for the enc x: "<< context.get_context_data(enc_ecd_x.params_id()).chain_depth()<<endl;

    synchronize_test_device();
    gettimeofday(&tstart1,NULL);

    PhantomCiphertext output = softmax_boot(enc_ecd_x,context,relin_keys,gal_keys,bootstrapper, 10,max);

    synchronize_test_device();
    gettimeofday(&tend1,NULL);
    cout <<"Modulus chain index for the output: "<< context.get_context_data(output.params_id()).chain_depth()<<endl;
    double softmax_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Softmax time = "<<softmax_time<<endl;

    cout <<"Decrypt + decode result: "<<endl;
    PhantomPlaintext plain_result;
    decryptor.decrypt(output, plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);
    cout <<result[0]<<" "<<result[slot_count/2]<<endl;
    ErrorStats errors(2e-2, 0);
    if (result.size() != slot_count) throw std::runtime_error("Unexpected slot count");
    for (std::size_t j = 0; j < slot_count; ++j)
        errors.add(result[j], output_pt[j/(slot_count/2)]);
    errors.report();


}
