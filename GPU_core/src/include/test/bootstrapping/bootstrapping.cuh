#pragma once
#include <random>

#include "include.cuh"
// #include "bootstrapping/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

void random_real(vector<double> &vec, size_t size)
{
  random_device rn;
  mt19937_64 rnd(rn());
  thread_local std::uniform_real_distribution<double> distribution(-1, 1);

  vec.reserve(size);

  for (size_t i = 0; i < size; i++)
  {
    vec[i] = distribution(rnd);
  }
}

void bootstrapping_test()
{
     // parameters setting
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;

    long logN = 16;
    long loge = 10;

    long logn = 15;
    long sparse_slots = (1 << logn);

    int logp = 51; // only 46, 51 the result is correct. 40 cannot
    int logq = 51; // only 46, 51 the result is correct. 40 cannot
    int log_special_prime = 60;

    int secret_key_hamming_weight = 192;

    // Calculation required
    int boot_level = 23;  // S2C 3 + C2S 3

    int remaining_level = 8;
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
    parms.set_sparse_slots(sparse_slots); //fix for EvalPoly 512(ct3)

    PhantomContext context(parms);
    const auto &modulus = context.first_context_data().parms().coeff_modulus();
    double q0 = ((double)modulus[0].value());
    double scale = q0;
    cout << "scale: " << scale << ", log(scale): " << log2(scale) << endl;

    int chain_depth = context.first_context_data().chain_depth() + 1;
    cout << "chain depth: " << chain_depth << endl;

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    // instance initialization

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    // PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);
    PhantomGaloisKey gal_keys;

    Encryptor encryptor(&context, &public_key);
    Decryptor decryptor(&context, &secret_key);
    // CKKSEncoder encoder(context);
    PhantomCKKSEncoder phantom_encoder(context);
    // repack the phantom encoder to SEAL style
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    size_t slot_count = encoder.slot_count();

    cout <<"Slot count: "<<slot_count<<endl;

    auto &context_data = context.key_context_data();
    auto coeff_modulus = context_data.parms().coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    double rlwe_ct_modulus = coeff_modulus[0].value();
    //cout <<"q0 = "<<rlwe_ct_modulus<<" log2(q0) = ";
    //cout <<log2(rlwe_ct_modulus)<<endl;

    //updated Sept 5: scale \Delta should be q/p.
    //double scale = rlwe_ct_modulus;
    cout <<"scale = "<<scale<<" log2(scale) = "<<log2(scale)<<endl;
    double spec_scale = pow(2.0, 46);

    //prepare for C2S, S2C and Modraise
    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &phantom_encoder, &relin_keys, &gal_keys, spec_scale);

    Bootstrapper bootstrapper(
    loge,
    logn,
    logN - 1,
    total_level,
    spec_scale,
    boundary_K,
    deg,
    scale_factor,
    inverse_deg,
    &ckks_evaluator);
    bootstrapper.set_final_scale(spec_scale);
    cout << "preparing bootstrapping..." << endl;
    bootstrapper.prepare_mod_polynomial();

    cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    gal_keys = secret_key.create_galois_keys(context);

    std::cout << "Galois key generated from steps vector." << endl;

    bootstrapper.slot_vec.push_back(logn);

    cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper.generate_LT_coefficient_3();

  duration<double> dur_enc, dur_boot;
  vector<PhantomCiphertext> output(1);
  // PnR pnr("bert_base");
  // pnr.load_all_data("layer_0");

 #pragma omp parallel num_threads(1)
 {


  vector<double> sparse(sparse_slots, 0.0);
  vector<double> input(slot_count, 0.0);
  vector<double> before(slot_count, 0.0);
  vector<double> after(slot_count, 0.0);

  random_real(sparse, sparse_slots);

  PhantomPlaintext plain;
  PhantomCiphertext cipher;

  // Create input cipher
  for (size_t i = 0; i < slot_count; i++)
  {
    input[i] = i;
  }

  ckks_evaluator.encoder.encode(input, spec_scale, plain);
  ckks_evaluator.encryptor.encrypt(plain, cipher);

  // Mod switch to the lowest level
  // for (int i = 0; i < total_level; i++)
  // {
  //   ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
  // }

  // Decrypt input cipher to obtain the original input
  ckks_evaluator.decryptor.decrypt(cipher, plain);
  ckks_evaluator.encoder.decode(plain, before);

  const int tid = omp_get_thread_num();
  // start = system_clock::now();

#pragma omp for schedule(static)
  for (int k = 0; k < 1; k++){
    auto s = system_clock::now();
    if (tid == 0){
      for (int i = 0; i < total_level; i++)
      {
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(cipher);
      }
      cout << "cipher level before bootstrapping: " << context.get_context_data(cipher.params_id()).chain_depth() << endl;
      PhantomCiphertext rtn;
      bootstrapper.bootstrap_3(rtn, cipher);
      output[k] = std::move(rtn);
      duration<double> d = system_clock::now() - s;
      dur_boot += d;
      // std::cout << "Thread: " << tid << " took " << d.count() << "s" << endl;
    }
    else{
      const int num = 0;
      vector<PhantomCiphertext> temp(num);
      for (int j = 0; j < num; j++){
        // Create input cipher
        for (size_t i = 0; i < slot_count; i++)
        {
          input[i] = sparse[i % sparse_slots];
        }

        ckks_evaluator.encoder.encode(input, scale, plain);
        ckks_evaluator.encryptor.encrypt(plain, temp[j]);
      }
      duration<double> d = system_clock::now() - s;
      dur_enc += d;
      // std::cout << "Thread: " << tid << " took " << d.count() << "s" << endl;
    }


    ckks_evaluator.decryptor.decrypt(output[0], plain);
    ckks_evaluator.encoder.decode(plain, after);

    double mean_err = 0;
    double std_err = 0;
    for (long i = 0; i < sparse_slots; i++)
    {
      // if (i < 10) std::cout << before[i] << " <----> " << after[i] << endl;
      mean_err += abs(before[i] - input[i]);
      std_err += (before[i] - input[i]) * (before[i] - input[i]);
    }
    mean_err /= sparse_slots;
    std_err = std::sqrt(std_err / sparse_slots);
    std::cout << "Mean absolute error: " << mean_err << endl;
    std::cout << "Standard deviation of error: " << std_err << endl;
  }
  bootstrapper.set_initial_scale(scale);
  cudaDeviceSynchronize();
 }
  std::cout << "Bootstrapping take:" << dur_boot.count() << "s" << endl;
  std::cout << "Encryption take:" << dur_enc.count() << "s" << endl;
  // duration<double> sec = system_clock::now() - start;
  // std::cout << "Bootstrapping took: " << sec.count() << "s" << endl;
  std::cout << "Return cipher level: " << output[0].coeff_modulus_size() << endl;


  // if (err != cudaSuccess) {
  //     std::cerr << "CUDA error after bootstrap: " << cudaGetErrorString(err) << std::endl;
  // }


}
