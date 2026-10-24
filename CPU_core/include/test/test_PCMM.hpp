#include <chrono>
#include "test_common.hpp"
using namespace std;
using namespace seal;

vector<vector<double>> MM(const vector<vector<double>> & x, const vector<vector<double>> & w){
    int x_row = x.size();
    int x_col = x[0].size();
    int w_row = w.size();
    int w_col = w[0].size();
   // cout <<"size of left matrix = "<<x_row<<" * "<<x_col<<endl;
   // cout <<"size of right matrix = "<<w_row<<" * "<<w_col<<endl;

    vector<vector<double>> output(x_row,vector<double>(w_col,0));
    for (int i = 0; i < x_row; ++i){
        for (int j = 0 ; j < w_col ; ++j){
            double sum = 0.0;
            for (int k = 0 ; k < x_col ; ++k){
                sum += (x[i][k]*w[k][j]);
            }
            output[i][j] = sum;
        }
    }

    return output;
}

vector<double> VM(const vector<double> & x, const vector<vector<double>> & w){
    int x_row = 1;
    int x_col = x.size();
    int w_row = w.size();
    int w_col = w[0].size();
    //cout <<"size of left matrix = "<<x_row<<" * "<<x_col<<endl;
    //cout <<"size of right matrix = "<<w_row<<" * "<<w_col<<endl;

    vector<double> output(w_col,0);

        for (int j = 0 ; j < w_col ; ++j){
            double sum = 0.0;
            for (int k = 0 ; k < x_col ; ++k){
                sum += (x[k]*w[k][j]);
            }
            output[j] = sum;
        }


    return output;
}

void vec_left_rot_test(){
    vector<double> input(50,0);
    cout <<"Input vector: "<<endl;
    for (int i = 0; i < 50; ++i){
        input[i] = i+1;
        cout <<input[i] <<" ";
    }
    cout <<endl;

    vector<double> output = Vec_left_rot(input,-10);
    cout <<"Output vector (left rotate -10): "<<endl;
    for (int i = 0; i < 50; ++i){
        cout <<output[i]<<" ";
    }
    cout <<endl;

    ErrorStats errors(0, 0);
    if (output.size() != input.size()) throw std::runtime_error("Unexpected rotation size");
    for (int i = 0; i < 50; ++i) errors.add(output[i], input[(i+40)%50]);
    errors.report();
}

void PCMM_test(){
    cout <<"Task: test Ct-Pt matrix multiplication in modified column packing: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    struct timeval tstart1, tend1;

    //construct input
    int num_row = 128;
    int num_col = 768;
    int k = 1;

    vector<vector<double>> input_x(num_row, vector<double>(num_col,0));

    for (int i = 0 ; i < num_row ; ++i){
        for (int j = 0 ; j < num_col ; ++j){
            input_x[i][j] = (double)(i+j)*0.002+0.02;
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

    vector<Ciphertext> enc_ecd_x = Modified_col_packing(input_x, num_row, num_col, k,
        scale, context,public_key);
/*
    for (int i = 0; i < enc_ecd_x.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/


    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 768;

    vector<vector<double>> w(W_row, vector<double>(W_col,1.0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.01+j*0.01+i*0.02;
        }
    }

    cout <<"Matrix W size = "<<W_row <<" * "<<W_col<<endl;

    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<w[i][j]<<" ";
        }
        cout <<endl;
    }

    vector<vector<double>> res = MM(input_x,w);
/*
    cout <<"Real result (10 * 10): "<<endl;
    for (int i = 0; i < 10; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res_pack = Modified_col_packing_pt(res,num_row, W_col, k, slot_count);

    for (int i = 0; i < res_pack.size(); ++i){
        cout <<i<<"-th vector: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<res_pack[i][ind]<<" ";
        }
        cout <<endl;
    }


    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_pt_mul = PCMM_modified_col_packing(enc_ecd_x, w,
        num_col, num_row, W_col, W_row, k, context,gal_keys);

    gettimeofday(&tend1,NULL);
    double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_pt_mul[0].parms_id())->chain_index()<<endl;

    vector<vector<double>> ct_res(ct_pt_mul.size());
    cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_pt_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
        //vector<double> result;
        encoder.decode(plain_result, ct_res[i]);

        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count-10 ; ind < slot_count ; ++ind){
            cout <<ct_res[i][ind]<<" ";
        }
        cout <<endl;
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




}

void PCMM_batch_test(){
    cout <<"Task: test Ct-Pt matrix multiplication in modified column packing + batching: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

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

/*
    for (int i = 0; i < enc_ecd_x.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/


    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 768;

    vector<vector<double>> w(W_row, vector<double>(W_col,1.0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.01+j*0.01+i*0.02;
        }
    }

    cout <<"Matrix W size = "<<W_row <<" * "<<W_col<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<w[i][j]<<" ";
        }
        cout <<endl;
    }
*/

    cout <<"--------------------------------------"<<endl;
    cout <<"Computing enc(X)W..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_pt_mul = PCMM_modified_col_packing_batch(enc_ecd_x, w,
        batch_size, num_col, num_row, W_col, W_row, k, context,gal_keys);

    gettimeofday(&tend1,NULL);
    double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_pt_mul[0].parms_id())->chain_index()<<endl;

     vector<vector<double>> res = MM(input_x[0],w);
/*
    cout <<"Real result (10 * 10): "<<endl;
    for (int i = 0; i < 10; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res_pack = Modified_col_packing_batch_pt(res, batch_size, num_row, W_col, k, slot_count);
/*
    for (int i = 0; i < res_pack.size(); ++i){
        cout <<i<<"-th vector: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<res_pack[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<res_pack[i][ind*batch_size]<<" ";
        }
        cout <<endl;
    }
*/

    vector<vector<double>> ct_res(ct_pt_mul.size());
 //   cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_pt_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
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

void PCMM_batch_complex_test(){
    cout <<"Task: test Ct-Pt matrix multiplication in modified column packing + batching + complex slot: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 60}));
    double scale = pow(2.0,40);

    SEALContext context(parms);

    cout <<"Set encryption parameters and print"<<endl;
    print_parameters(context);

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Decryptor decryptor(context, secret_key);
    CKKSEncoder encoder(context);
    Evaluator evaluator(context, encoder);
    size_t slot_count = encoder.slot_count();

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

/*
    for (int i = 0; i < enc_ecd_x.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(enc_ecd_x[i], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        cout <<i<<"-th ciphertext: ";
        for (int ind = 0 ; ind < 10 ; ++ind){
            cout <<result[ind]<<" ";
        }
        cout <<endl;
    }
*/


    //construct plaintext matrix
    int W_row = num_col;
    int W_col = 3072;

    vector<vector<double>> w(W_row, vector<double>(W_col,1.0));

    for (int i = 0 ; i < W_row ; ++i){
        for (int j = 0 ; j < W_col ; ++j){
            w[i][j] = 0.01+j*0.01+i*0.02;
        }
    }

    cout <<"Matrix W size = "<<W_row <<" * "<<W_col<<endl;
/*
    cout <<"First 10 elements in the first 10 rows: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout<<w[i][j]<<" ";
        }
        cout <<endl;
    }
*/

    cout <<"--------------------------------------"<<endl;
    cout <<"Computing enc(X)W..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ct_pt_mul = PCMM_modified_col_packing_batch_complex(enc_ecd_x, w,
        batch_size, num_col, num_row, W_col, W_row, k, context,gal_keys);

    gettimeofday(&tend1,NULL);
    double ct_pt_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Ct-Pt matrix multiplication time = "<<ct_pt_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(ct_pt_mul[0].parms_id())->chain_index()<<endl;

     vector<vector<double>> res = MM(input_x[0],w);
/*
    cout <<"Real result (10 * 10): "<<endl;
    for (int i = 0; i < 10; ++i){
        for (int j = 0 ; j < 10 ; ++j){
            cout <<res[i][j]<<" ";
        }
        cout <<endl;
    }
*/
    vector<vector<double>> res_pack = Modified_col_packing_batch_pt(res, batch_size, num_row, W_col, k, slot_count);
/*
    for (int i = 0; i < res_pack.size(); ++i){
        cout <<i<<"-th vector: ";
        for (int ind = 0 ; ind < 5 ; ++ind){
            cout <<res_pack[i][ind*batch_size]<<" ";
        }
        cout <<" ... ";
        for (int ind = slot_count/batch_size-5 ; ind < slot_count/batch_size; ++ind){
            cout <<res_pack[i][ind*batch_size]<<" ";
        }
        cout <<endl;
    }

*/
    vector<vector<complex<double>>> ct_res(ct_pt_mul.size());
 //   cout <<"Decrypt + decode result: "<<endl;
    //decrypt and decode
    for (int i = 0; i < ct_pt_mul.size(); ++i){
        Plaintext plain_result;
        decryptor.decrypt(ct_pt_mul[i], plain_result);
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
    ErrorStats errors(1e-3);
    for (int i = 0; i < ct_res.size(); ++i){
        for (int j = 0 ; j < slot_count ; ++j){
            errors.add(ct_res[i][j].real(), res_pack[2*i][j]);
            errors.add(ct_res[i][j].imag(), res_pack[2*i+1][j]);
        }
    }
    errors.report();
    }
    cout <<"=========================TEST END======================="<<endl<<endl;

}

void PCMM_batch_reference_test() {
        constexpr int batch = 16, rows = 128, columns = 32, outputs = 64;
        constexpr double scale = 1099511627776.0;
        EncryptionParameters parameters(scheme_type::ckks);
        parameters.set_poly_modulus_degree(65536);
        parameters.set_coeff_modulus(CoeffModulus::Create(65536, {60, 40, 40, 60}));
        SEALContext context(parameters);
        KeyGenerator keygen(context);
        PublicKey public_key;
        keygen.create_public_key(public_key);
        GaloisKeys rotation_keys;
        keygen.create_galois_keys(std::vector<int>{2048, 4096, 6144, 8192, 16384, 24576}, rotation_keys);
        Decryptor decryptor(context, keygen.secret_key());
        CKKSEncoder encoder(context);

        std::vector<std::vector<std::vector<double>>> input(
            batch, std::vector<std::vector<double>>(rows, std::vector<double>(columns)));
        std::vector<std::vector<double>> weights(columns, std::vector<double>(outputs));
        for (int b = 0; b < batch; ++b)
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < columns; ++c)
                    input[b][r][c] = 0.5 * std::sin(0.1 * (b + 1) + 0.013 * r + 0.017 * c);
        for (int c = 0; c < columns; ++c)
            for (int o = 0; o < outputs; ++o)
                weights[c][o] = 0.05 * std::cos(0.07 * c + 0.11 * o);

        auto encrypted = Modified_col_packing_batch(input, batch, rows, columns, 1, scale, context, public_key);
        const auto start = std::chrono::steady_clock::now();
        auto result = PCMM_modified_col_packing_batch_complex(
            encrypted, weights, batch, columns, rows, outputs, columns, 1, context, rotation_keys);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        const std::size_t slots = encoder.slot_count();
        const std::size_t tile = slots / (batch * rows);
        if (result.size() * 2 * tile != outputs)
            throw std::runtime_error("Unexpected output shape");
        ErrorStats errors(1e-4, 0);
        for (std::size_t i = 0; i < result.size(); ++i) {
            Plaintext plaintext;
            std::vector<std::complex<double>> decoded;
            decryptor.decrypt(result[i], plaintext);
            encoder.decode(plaintext, decoded);
            if (decoded.size() != slots)
                throw std::runtime_error("Unexpected slot count");
            for (std::size_t s = 0; s < slots; ++s) {
                const int b = s % batch;
                const int r = (s / batch) % rows;
                for (int component = 0; component < 2; ++component) {
                    const int o = (2 * i + component) * tile + s / (batch * rows);
                    double reference = 0.0;
                    for (int c = 0; c < columns; ++c)
                        reference += input[b][r][c] * weights[c][o];
                    const double value = component == 0 ? decoded[s].real() : decoded[s].imag();
                    errors.add(value, reference);
                }
            }
        }
    std::cout << "PCMM time = " << seconds << std::endl;
    errors.report();
}
