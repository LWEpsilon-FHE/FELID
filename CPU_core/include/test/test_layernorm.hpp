#include "test_common.hpp"
#include "Bootstrapper.h"

using namespace std;
using namespace seal;

void Layernorm_test(){
    cout <<"Task: test Layernorm in interleaved column packing: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));
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
                input_x[ind][i][j] = (double)(j)*0.1+0.2;
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

    vector<double> gamma(num_col), beta(num_col);
    for (int j = 0; j < num_col; ++j) {
        gamma[j] = 0.05 + 0.0001*j;
        beta[j] = 0.1 + 0.0002*j;
    }

    vector<vector<double>> output_pt = layernorm_pt(input_x[0], input_x[0], num_row, num_col, gamma, beta);
/*
    cout <<"First 10 elements in the first 10 rows in plain_result: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<output_pt[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res_pack = Modified_col_packing_batch_pt(output_pt,batch_size,num_row, num_col, k, slot_count);
/*
    cout <<"packed pt result: "<<endl;
    for (int i = 0; i < res_pack.size(); ++i){
        for (int ind = 0; ind < 10*batch_size ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<endl;
    }
*/


    //encode + encrypt


    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);


    cout <<"Modulus chain index for the enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> output = layernorm_v2(enc_ecd_x,enc_ecd_x,gamma,beta,768.0,num_row,num_col,batch_size,context,relin_keys,gal_keys);

    gettimeofday(&tend1,NULL);
    double ln_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Layernorm time = "<<ln_time<<endl;

    vector<vector<double>> ct_res(output.size());
 //   cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
        /*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<ct_res[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<ct_res[i][ind*batch_size]<<" ";
        }
        cout <<endl;
        */
    }

    //compute error

    {
    ErrorStats errors(5e-3);
    for (int i = 0; i < ct_res.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res[i][j], res_pack[i][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;

/*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt

    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10*batch_size ; ind += batch_size){
            cout <<result[ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/
}

void Layernorm_boot_selfoutput_test(){
    cout <<"Task: test Layernorm_boot_selfoutput in interleaved column packing: "<<endl;

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

    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    size_t slot_count = encoder.slot_count();
    cout <<"slot count = "<<slot_count<<endl;

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
      context,
      keygen,
      encoder,
      encryptor,
      decryptor,
      evaluator,
      relin_keys,
      gal_keys);

  cout << "Generating Optimal Minimax Polynomials..." << endl;
  bootstrapper.prepare_mod_polynomial();

  cout << "Adding Bootstrapping Keys..." << endl;
  vector<int> gal_steps_vector;
  gal_steps_vector.push_back(0);
  for (int i = 0; i < logN - 1; i++) {
    gal_steps_vector.push_back((1 << i));
  }
  bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

  keygen.create_galois_keys(gal_steps_vector, gal_keys);

  bootstrapper.slot_vec.push_back(logn);

  cout << "Generating Linear Transformation Coefficients..." << endl;
  bootstrapper.generate_LT_coefficient_3();


    struct timeval tstart1, tend1;

    //construct input
    int num_row = 128;
    int num_col = 32;
    int k = 1;
    int batch_size = 16;

    vector<vector<vector<double>>> input_x(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                input_x[ind][i][j] = (double)(j)*0.1+0.2;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<input_x[0][i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<double> gamma(num_col), beta(num_col);
    for (int j = 0; j < num_col; ++j) {
        gamma[j] = 0.05 + 0.0001*j;
        beta[j] = 0.1 + 0.0002*j;
    }




    //encode + encrypt


    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    for (int i = 0; i < enc_ecd_x.size(); ++i){
        for (int j = 0 ; j < boot_level+6 ; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }


    cout <<"Modulus chain index for the enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;  //6

    vector<vector<double>> res_pack = layernorm_pt(input_x[0], input_x[0], num_row, num_col, gamma, beta);

    vector<vector<vector<double>>>().swap(input_x);
/*
    cout <<"First 10 elements in the first 10 rows in plain_result: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<output_pt[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    res_pack = Modified_col_packing_batch_pt(res_pack,batch_size,num_row, num_col, k, slot_count);
/*
    cout <<"packed pt result: "<<endl;
    for (int i = 0; i < res_pack.size(); ++i){
        for (int ind = 0; ind < 10*batch_size ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<endl;
    }
*/

    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> output = layernorm_boot_self(enc_ecd_x,enc_ecd_x,gamma,beta,4.0,num_row,num_col,batch_size,
        context,relin_keys,gal_keys,bootstrapper);

    gettimeofday(&tend1,NULL);
    double ln_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Layernorm time = "<<ln_time<<endl;

    vector<vector<double>> ct_res(output.size());
 //   cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);
        /*
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<ct_res[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<ct_res[i][ind*batch_size]<<" ";
        }
        cout <<endl;
        */
    }

    //compute error

    {
    ErrorStats errors(1e-2, 0);
    for (int i = 0; i < ct_res.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res[i][j], res_pack[i][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;

/*
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt

    for (int i = 0; i < output.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(output[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10*batch_size ; ind += batch_size){
            cout <<result[ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-(10*batch_size) ; ind < slot_count ; ind += batch_size){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/
}









void Layernorm_affine_test() {
    constexpr int batch = 16, rows = 128, columns = 32;
    vector<int> bits(22, 40);
    bits.front() = bits.back() = 60;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(65536);
    parms.set_coeff_modulus(CoeffModulus::Create(65536, bits));
    SEALContext context(parms);
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin;
    keygen.create_relin_keys(relin);
    GaloisKeys rotations;
    keygen.create_galois_keys(rotations);
    CKKSEncoder encoder(context);
    Decryptor decryptor(context, keygen.secret_key());
    vector<vector<vector<double>>> input(batch,
        vector<vector<double>>(rows, vector<double>(columns)));
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < columns; ++c)
                input[b][r][c] = 0.2 + 0.1*c + 0.01*b + 0.001*r;
    auto x = Modified_col_packing_batch(input, batch, rows, columns, 1,
                                        pow(2.0, 40), context, public_key);
    vector<double> gamma(columns, 0.1), zero(columns, 0), bias(columns), gain(columns);
    for (int c = 0; c < columns; ++c) {
        bias[c] = 0.001*c;
        gain[c] = 0.05 + 0.001*c;
    }
    auto base = layernorm_v2(x, x, gamma, zero, 0.341,
                            rows, columns, batch, context, relin, rotations);
    auto biased = layernorm_v2(x, x, gamma, bias, 0.341,
                              rows, columns, batch, context, relin, rotations);
    auto scaled = layernorm_v2(x, x, gain, zero, 0.341,
                              rows, columns, batch, context, relin, rotations);
    if (base.size() != 2 || biased.size() != base.size() || scaled.size() != base.size())
        throw std::runtime_error("Unexpected LayerNorm output shape");
    ErrorStats bias_errors(1e-5, 0), gain_errors(1e-5, 0);
    for (std::size_t i = 0; i < base.size(); ++i) {
        Plaintext plain;
        vector<double> a, b, g;
        decryptor.decrypt(base[i], plain); encoder.decode(plain, a);
        decryptor.decrypt(biased[i], plain); encoder.decode(plain, b);
        decryptor.decrypt(scaled[i], plain); encoder.decode(plain, g);
        if (a.size() != encoder.slot_count() || b.size() != a.size() || g.size() != a.size())
            throw std::runtime_error("Unexpected LayerNorm slot count");
        for (std::size_t s = 0; s < a.size(); ++s) {
            const std::size_t c = i*16 + s/(batch*rows);
            bias_errors.add(b[s] - a[s], bias[c]);
            gain_errors.add(g[s], a[s]*gain[c]/0.1);
        }
    }
    bias_errors.report();
    gain_errors.report();
}
