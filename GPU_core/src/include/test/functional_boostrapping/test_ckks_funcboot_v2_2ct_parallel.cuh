#include "include.cuh"

using namespace std;
using namespace phantom;
using namespace fhe;


void funcboot_v2_2ct_parallel_test(){
    cout <<"------------------------------------------"<<endl;
    cout <<"Parallel Task: test Function bootstrapping (v2) algorithms with 2 slot-encoded inputs in CKKS"<<endl;
    cout <<"Single Thread TEST. "<<endl;
    //bootstrapping parameters
    long boundary_K = 25;
  long deg = 59;
  long scale_factor = 2;
  long inverse_deg = 1;

  long logN = 16;
  long loge = 10;

  long logn = 15;
  long sparse_slots = (1 << logn);

  int logp = 46; // only 46, 51 the result is correct. 40 cannot
  int logq = 46; // only 46, 51 the result is correct. 40 cannot
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

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);


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
    //prepare for C2S, S2C and Modraise
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

        //read coeff from txt file
    vector<complex<double>> exp_coeff(coeff25);

    vector<complex<double>> lut_coeff(512,0);

    cudaDeviceSynchronize();


    ifstream infile;

    infile.open("Hermite4.txt");
    if(!infile.is_open()){
        cout <<"cannot open file Hermite4.txt. "<<endl;
    }
    for (int i = 0; i < 512; ++i){
        double real, imag;
        infile >> real;
        infile >> imag;
        lut_coeff[i].real(real);
        lut_coeff[i].imag(imag);
        //cout <<exp_coeff[i]<<endl;
    }
    infile.close();
    //cout <<exp_coeff[0]<<", "<<exp_coeff[58]<<endl;
    //cout <<lut_coeff[0]<<", "<<lut_coeff[511]<<endl;

    vector<double> input1(slot_count,64.0);

    PhantomPlaintext pt1;
    encoder.encode(input1,scale,pt1);

    PhantomCiphertext ct1;
    encryptor.encrypt(pt1, ct1);

    vector<double> input2(slot_count,128.0);
    encoder.encode(input2,scale,pt1);
    PhantomCiphertext ct2;
    encryptor.encrypt(pt1,ct2);

    //test Decrypt and decode the input:
    PhantomPlaintext test_pt1;
    decryptor.decrypt(ct1,test_pt1);

    vector<double> test_output1;
    encoder.decode(test_pt1,test_output1);
    cout <<"Decrypt + decode result of input1: ";
    for (int i = 0; i < 10; ++i){
        cout <<test_output1[i]<<" ";
    }
    cout <<endl;

    // while(ct1.coeff_modulus_size() != 5) evaluator.mod_switch_to_next_inplace(ct1);
    // evaluator.mod_switch_to_inplace(ct2, ct1.params_id());

    cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.params_id()).chain_depth()<<endl;
    cout <<"    Modulus chain index for ct2: "<< context.get_context_data(ct2.params_id()).chain_depth()<<endl;



    // can I parallel this?
    vector<PhantomCiphertext> ct_output;

    auto time_start = chrono::high_resolution_clock::now();
    ct_output = FuncBoot_slot_v2_2ct_parallel(ct1, ct2, lut_coeff, exp_coeff, relin_keys, bootstrapper,
        poly_modulus_degree, slot_count, context, secret_key);
    auto time_end = chrono::high_resolution_clock::now();
    auto time_diff = duration_cast<duration<double>>(time_end - time_start).count();
    // cout <<"Functional Bootstrapping (v2) with 2 slot-encoded inputs takes time: "<<time_diff<<" seconds."<<endl;


        // cudaDeviceSynchronize();



//     auto time_start = chrono::high_resolution_clock::now();
//     vector<PhantomCiphertext> ct_output = FuncBoot_slot_v2_2ct_parallel(ct1, ct2, lut_coeff, exp_coeff, relin_keys, bootstrapper,
//         poly_modulus_degree, slot_count, context, secret_key);
//     auto time_end = chrono::high_resolution_clock::now();
//     auto time_diff = duration_cast<duration<double>>(time_end - time_start).count();
//     cout <<"Functional Bootstrapping (v2) with 2 slot-encoded inputs takes time: "<<time_diff<<" seconds."<<endl;

//    // cout <<"    Modulus chain index for ct_output: "<< context.get_context_data(ct_output.parms_id())->chain_index()<<endl;
//    //cout <<"    scale of output ct: "<<ct_output.scale()<<" log2(scale): "<<log2(ct_output.scale())<<endl;

//     // can I copy everything?
//     auto time_start_copy = chrono::high_resolution_clock::now();
//     vector<PhantomCiphertext> ct_output_copy = FuncBoot_slot_v2_2ct_parallel(ct1_copy, ct2_copy, lut_coeff_copy, exp_coeff_copy, relin_keys_copy, bootstrapper_copy,
//         poly_modulus_degree, slot_count_copy, context_copy, secret_key_copy);
//     auto time_end_copy = chrono::high_resolution_clock::now();
//     auto time_diff_copy = duration_cast<duration<double>>(time_end_copy - time_start_copy).count();
//     cout <<"[Copy Context] Functional Bootstrapping (v2) with 2 slot-encoded inputs takes time: "<<time_diff_copy<<" seconds."<<endl;




    //test decrypt ct_output
    PhantomPlaintext pt_output;
    decryptor.decrypt(ct_output[0],pt_output);
    vector<complex<double>> dcd_output;
    encoder.decode(pt_output,dcd_output);
    cout <<"Decrypt result of ct_output[0]: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;

    decryptor.decrypt(ct_output[1],pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"Decrypt result of ct_output[1]: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;


    cout <<endl;

}


// cpu version
// void funcboot_v2_2ct_test(){
//     cout <<"------------------------------------------"<<endl;
//     cout <<"Task: test Function bootstrapping (v2) algorithms with 2 slot-encoded inputs in CKKS"<<endl;
//     cout <<"Single Thread TEST. "<<endl;
//     //bootstrapping parameters
//     long boundary_K = 25;
//   long deg = 59;
//   long scale_factor = 2;
//   long inverse_deg = 1;

//   long logN = 16;
//   long loge = 10;

//   long logn = 15;
//   long sparse_slots = (1 << logn);

//   int logp = 40;
//   int logq = 40;
//   int log_special_prime = 60;

//   int secret_key_hamming_weight = 192;

//   // Calculation required
//   int boot_level = 23;  // S2C 3 + C2S 3

//   int remaining_level = 1;
//   int total_level = remaining_level + boot_level;

//   vector<int> coeff_bit_vec;
//   coeff_bit_vec.push_back(logq);
//   for (int i = 0; i < remaining_level; i++) {
//     coeff_bit_vec.push_back(logp);
//   }
//   for (int i = 0; i < boot_level; i++) {
//     coeff_bit_vec.push_back(logq);
//   }
//   coeff_bit_vec.push_back(log_special_prime);


//     EncryptionParameters parms(scheme_type::ckks);

//     size_t poly_modulus_degree = (size_t)(1 << logN);
//     parms.set_poly_modulus_degree(poly_modulus_degree);
//     parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
//     parms.set_secret_key_hamming_weight(secret_key_hamming_weight);

//     SEALContext context(parms, true, sec_level_type::none);

//     const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
//     double q0 = ((double)modulus[0].value());

//     double scale = q0;

//     cout <<"Set encryption parameters and print"<<endl;
//     print_parameters(context);

//     KeyGenerator keygen(context);
//     SecretKey secret_key = keygen.secret_key();
//     PublicKey public_key;
//     keygen.create_public_key(public_key);
//     RelinKeys relin_keys;
//     keygen.create_relin_keys(relin_keys);

//     GaloisKeys gal_keys;
//     //keygen.create_galois_keys(gal_keys);

//     Encryptor encryptor(context, public_key);
//     Decryptor decryptor(context, secret_key);
//     CKKSEncoder encoder(context);
//     Evaluator evaluator(context, encoder);
//     size_t slot_count = (size_t)(1 << logn);
//     cout <<"Slot count: "<<slot_count<<endl;

//     auto &context_data = *context.key_context_data();
//     auto coeff_modulus = context_data.parms().coeff_modulus();
//     size_t coeff_modulus_size = coeff_modulus.size();
//     double rlwe_ct_modulus = coeff_modulus[0].value();
//     //cout <<"q0 = "<<rlwe_ct_modulus<<" log2(q0) = ";
//     //cout <<log2(rlwe_ct_modulus)<<endl;

//     //updated Sept 5: scale \Delta should be q/p.
//     //double scale = rlwe_ct_modulus;
//     cout <<"scale = "<<scale<<" log2(scale) = "<<log2(scale)<<endl;
//     //prepare for C2S, S2C and Modraise
//     Bootstrapper bootstrapper(
//       loge,
//       logn,
//       logN - 1,
//       total_level,
//       scale,
//       boundary_K,
//       deg,
//       scale_factor,
//       inverse_deg,
//       context,
//       keygen,
//       encoder,
//       encryptor,
//       decryptor,
//       evaluator,
//       relin_keys,
//       gal_keys);

//     bootstrapper.prepare_mod_polynomial();

//     vector<int> gal_steps_vector;
//     gal_steps_vector.push_back(0);
//     for (int i = 0; i < logN - 1; i++) {
//         gal_steps_vector.push_back((1 << i));
//     }
//     bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

//     keygen.create_galois_keys(gal_steps_vector, gal_keys);

//     bootstrapper.slot_vec.push_back(logn);

//   //cout << "Generating Linear Transformation Coefficients..." << endl;
//   bootstrapper.generate_LT_coefficient_3();

//         //read coeff from txt file
//     vector<complex<double>> exp_coeff(coeff25);

//     vector<complex<double>> lut_coeff(512,0);


//     ifstream infile;

//     infile.open("Hermite4.txt");
//     if(!infile.is_open()){
//         cout <<"cannot open file Hermite4.txt. "<<endl;
//     }
//     for (int i = 0; i < 512; ++i){
//         double real, imag;
//         infile >> real;
//         infile >> imag;
//         lut_coeff[i].real(real);
//         lut_coeff[i].imag(imag);
//         //cout <<exp_coeff[i]<<endl;
//     }
//     infile.close();
//     //cout <<exp_coeff[0]<<", "<<exp_coeff[58]<<endl;
//     //cout <<lut_coeff[0]<<", "<<lut_coeff[511]<<endl;

//     vector<double> input1(slot_count,64.0);

//     Plaintext pt1;
//     encoder.encode(input1,scale,pt1);

//     Ciphertext ct1;
//     encryptor.encrypt(pt1, ct1);

//     vector<double> input2(slot_count,128.0);
//     encoder.encode(input2,scale,pt1);
//     Ciphertext ct2;
//     encryptor.encrypt(pt1,ct2);

//     //test Decrypt and decode the input:
//     Plaintext test_pt1;
//     decryptor.decrypt(ct1,test_pt1);

//     vector<double> test_output1;
//     encoder.decode(test_pt1,test_output1);
//     cout <<"Decrypt + decode result of input1: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<test_output1[i]<<" ";
//     }
//     cout <<endl;

//     while(ct1.coeff_modulus_size() != 5) evaluator.mod_switch_to_next_inplace(ct1);
//     evaluator.mod_switch_to_inplace(ct2, ct1.parms_id());

//     cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.parms_id())->chain_index()<<endl;
//     cout <<"    Modulus chain index for ct2: "<< context.get_context_data(ct2.parms_id())->chain_index()<<endl;



//     vector<Ciphertext> ct_output = FuncBoot_slot_v2_2ct(ct1, ct2, lut_coeff, exp_coeff, relin_keys, bootstrapper,
//         poly_modulus_degree, slot_count, context, secret_key);

//    // cout <<"    Modulus chain index for ct_output: "<< context.get_context_data(ct_output.parms_id())->chain_index()<<endl;
//    //cout <<"    scale of output ct: "<<ct_output.scale()<<" log2(scale): "<<log2(ct_output.scale())<<endl;

//     //test decrypt ct_output
//     Plaintext pt_output;
//     decryptor.decrypt(ct_output[0],pt_output);
//     vector<complex<double>> dcd_output;
//     encoder.decode(pt_output,dcd_output);
//     cout <<"Decrypt result of ct_output[0]: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(ct_output[1],pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"Decrypt result of ct_output[1]: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;


//     cout <<endl;

// }
