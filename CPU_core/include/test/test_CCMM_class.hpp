#include "test_common.hpp"
using namespace std;
using namespace seal;

void CCMM_class_test(){
    cout <<"Task: test Ct-Ct matrix multiplication in classifier: "<<endl;

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree,
        //{60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,40, 40, 40, 40, 40, 40,40, 40,40,60}));
        {60, 40, 40, 40, 40, 60}));
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

    int num_row = 128;
    int num_col = 768;
    int k = 1;
    int batch_size = 16;

   vector<vector<double>> C(num_col, vector<double>(num_col,0));

        for (int i = 0 ; i < num_col ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                C[i][j] = (double)(i)*0.002+0.001;
            }
        }

    vector<vector<vector<double>>> input_x(batch_size,
        vector<vector<double>>(num_row, vector<double>(num_col,0)));

    for (int ind = 0 ; ind < batch_size ; ++ind){
        for (int i = 0 ; i < num_row ; ++i){
            for (int j = 0 ; j < num_col ; ++j){
                input_x[ind][i][j] = (double)(i+j)*0.002+0.001;
            }
        }
    }

    cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;

    vector<Ciphertext> enc_ecd_x = Modified_col_packing_batch(input_x, batch_size,
     num_row, num_col, k, scale, context,public_key);

    cout <<"Modulus chain index for enc x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;
    cout <<"Size of enc x: "<<enc_ecd_x.size()<<endl;



    cout <<"Matrix C size = "<<num_col <<" * "<<num_col<<", k = "<<k<<", batch size = "<<batch_size<<endl;



    vector<Ciphertext> enc_ecd_C = Class_1_packing_batch(C, batch_size,
     num_row, num_col, num_col, k, scale, context,public_key);

    for (int i = 0; i < enc_ecd_C.size(); ++i){
        evaluator.mod_switch_to_next_inplace(enc_ecd_C[i]);
    }

    cout <<"Modulus chain index for enc C: "<< context.get_context_data(enc_ecd_C[0].parms_id())->chain_index()<<endl;
    cout <<"Size of enc C: "<<enc_ecd_C.size()<<endl;



    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(X)Enc(C)..."<<endl;
    gettimeofday(&tstart1,NULL);

    vector<Ciphertext> ext_ecd_x = ext_copy_X(enc_ecd_x,batch_size, num_col,num_row,context,gal_keys);

    Ciphertext output = CCMM_class_1_batch(ext_ecd_x, enc_ecd_C, batch_size,
        num_col, num_row, num_col, num_col, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Enc(X)*Enc(C) time = "<<ct_ct_matrix_mul_time<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(output.parms_id())->chain_index()<<endl;


    //pt result
    vector<double> XC= VM(input_x[0][0],C);
/*
    cout <<"XC: First 10 elements: "<<endl;
    for (int i = 0 ; i < 10 ; ++i){
       cout<<XC[i]<<" ";
    }
    cout <<endl;
*/

 //   cout <<"Decrypt + decode result: "<<endl;
    vector<double> ct_res(output.size());
    //decrypt and decode

    Plaintext plain_result;
    decryptor.decrypt(output, plain_result);
    encoder.decode(plain_result, ct_res);
/*
    for (int ind = 0 ; ind < 10*batch_size ; ind += batch_size){
        cout <<ct_res[ind]<<" ";
    }

    cout <<endl;
*/

    //compute error

    {
    ErrorStats errors(1e-3);
    for (int j = 0 ; j < num_col*batch_size ; ++j){
        errors.add(ct_res[j], XC[j/batch_size]);
        }
    errors.report();
    }

    vector<vector<double>> C2(num_col, vector<double>(2,0));

    for (int i = 0 ; i < num_col ; ++i){
        for (int j = 0 ; j < 2 ; ++j){
            C2[i][j] = (double)(i)*0.002+0.001+0.0001*j;
        }
    }

    Ciphertext enc_ecd_C2 = Class_2_packing_batch(C2,batch_size,num_col,2,k,scale, context,public_key);
    evaluator.mod_switch_to_inplace(enc_ecd_C2,output.parms_id());

    cout <<"--------------------------------------"<<endl;
    cout<<"Computing Enc(XC)Enc(C2)..."<<endl;
    gettimeofday(&tstart1,NULL);

    Ciphertext output2 = CCMM_class_2_batch(output, enc_ecd_C2, batch_size,
        num_col, 1, 2, num_col, k, context,gal_keys,relin_keys);

    gettimeofday(&tend1,NULL);
    double ct_ct_matrix_mul_time2 = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Finished! Enc(XC)*Enc(C2) time = "<<ct_ct_matrix_mul_time2<<endl;
    cout <<"Modulus chain index for the result: "<< context.get_context_data(output2.parms_id())->chain_index()<<endl;

    decryptor.decrypt(output2, plain_result);
    encoder.decode(plain_result, ct_res);
    double ct_res_1 = ct_res[0];
    double ct_res_2 = ct_res[slot_count/2];
    cout <<"ct result = "<<ct_res_1 <<" "<<ct_res_2<<endl;


    vector<double> XCC2= VM(XC,C2);
    cout <<"pt result = "<<XCC2[0] << " "<<XCC2[1]<<endl;
    ErrorStats head_errors;
    if (ct_res.size() != slot_count) throw std::runtime_error("Unexpected slot count");
    for (int c = 0; c < 2; ++c)
        for (int b = 0; b < batch_size; ++b)
            head_errors.add(ct_res[c*(slot_count/2)+b], XCC2[c]);
    head_errors.report();


    cout <<"=========================TEST END======================="<<endl<<endl;





}
