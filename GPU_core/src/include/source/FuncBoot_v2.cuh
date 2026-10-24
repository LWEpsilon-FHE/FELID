#pragma once
#include <chrono>
using namespace chrono;
#include "include.cuh"
// #include "Bootstrapper.h"


vector<PhantomCiphertext> FuncBoot_slot_v2(const PhantomCiphertext & ct, const vector<complex<double>> & lut_coeff,
    const vector<complex<double>> & exp_coeff, const PhantomRelinKey &relin_keys, Bootstrapper& bootstrapper,
    int poly_modulus_degree, int slot_count, PhantomContext &context, PhantomSecretKey & secret_key){

    cout <<"------------Functional bootstrapping--------------------"<<endl;


    bootstrapper.initial_scale = ct.scale();
    double scale = ct.scale();
   // cout <<"    scale of input ct: "<<ct.scale()<<" log2(scale): "<<log2(ct.scale())<<endl;
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);

    //for test
    Decryptor decryptor(&context, &secret_key);

    PhantomPlaintext pt_output;
    vector<complex<double>> dcd_output;

    PhantomCiphertext input1 = ct;
    cout <<endl;


    //cout <<poly_modulus_degree<<" "<<slot_count<<endl;




    //input1 = input1/p
    cout <<"0. input1 = input1/512. input2 = input2/512"<<endl;
    cout << "input1 scale before division: "<<input1.scale()<<" log2(scale): "<<log2(input1.scale())<<endl;
    PhantomPlaintext ptp;
    encoder.encode(1/512.0, input1.params_id(), input1.scale(),ptp);
    evaluator.multiply_plain_inplace(input1,ptp);
    evaluator.rescale_to_next_inplace(input1);

    // evaluator.multiply_plain_inplace(input2,ptp);
    // evaluator.rescale_to_next_inplace(input2);

    cout <<"    Modulus chain index for input ct: "<< context.get_context_data(input1.params_id()).chain_depth()<<", ";
    cout <<"scale of input ct: "<<input1.scale()<<" log2(scale): "<<log2(input1.scale())<<endl;

    // input1.scale()=scale;
    // input2.scale()=scale;

    decryptor.decrypt(input1,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of input1: ";
    for (int i = 0; i < 15; ++i){
        cout <<dcd_output[i].real()<<" ";
    }
    cout <<endl;

    // decryptor.decrypt(input2,pt_output);
    // encoder.decode(pt_output,dcd_output);
    // cout <<"    Decrypt result of input2: ";
    // for (int i = 0; i < 15; ++i){
    //     cout <<dcd_output[i].real()<<" ";
    // }
    // cout <<endl;
    // cout<<endl;

    //ct1 = S2C(ct), level = 3
    cout <<"1. ct1 = S2C(input1,input2). "<<endl;

   // cout <<"    Modulus chain index for input ct: "<< context.get_context_data(input1.parms_id())->chain_index()<<", ";
   // cout <<"scale of input ct: "<<input1.scale()<<" log2(scale): "<<log2(input1.scale())<<endl;
    //S2C start
    auto t0 = chrono::high_resolution_clock::now();


    //Ciphertext input2 = ct;
    PhantomCiphertext ct1;

    bootstrapper.slottocoeff_3(ct1,input1);

    ct1.scale()=scale;

    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> t_s2c = t1 - t0;

    cout <<"    S2C. time = "<<t_s2c.count()<<" s. "<<endl;

    cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.params_id()).chain_depth()<<", ";
    cout <<"scale of ct1: "<<ct1.scale()<<" log2(scale): "<<log2(ct1.scale())<<endl;


    decryptor.decrypt(ct1,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of S2C: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<" ...... ";
    cout <<endl;
    for (int i = std::max(0, int(dcd_output.size())-10); i < int(dcd_output.size()); ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;
    cout <<endl;

    //mod switch to lowest level

    while(ct1.coeff_modulus_size() != 1) evaluator.mod_switch_to_next_inplace(ct1);

//    cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.parms_id())->chain_index()<<endl;


    //ct2 = ct1 = ModRaise(ct1)
    cout <<"2. ct1 = ModRaise(ct1). "<<endl;
    auto t2 = chrono::high_resolution_clock::now();

    //ModRaise_inplace(ct1,context);
    // bootstrapper.modraise_inplace(ct1);
    ModRaise_inplace(ct1, context);

    auto t3 = chrono::high_resolution_clock::now();
    chrono::duration<double> t_mr = t3 - t2;

    cout <<"    ModRaise. time = "<<t_mr.count()<<" s. "<<endl;

    cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.params_id()).chain_depth()<<", ";
    cout <<"scale of ct1: "<<ct1.scale()<<" log2(scale): "<<log2(ct1.scale())<<endl;

    decryptor.decrypt(ct1,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of ModRaise: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<" ...... ";
    cout <<endl;
    for (int i = std::max(0, int(dcd_output.size())-10); i < int(dcd_output.size()); ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;
    cout <<endl;

    //ct3 = C2S(ct2), level = 3
    cout <<"3. ct2 = C2S(ct1)/25.0. "<<endl;

    //C2S start
    auto t4 = chrono::high_resolution_clock::now();

    PhantomCiphertext ct2, dct2;

    bootstrapper.coefftoslot_full_3(ct2,dct2,ct1);

    ct2.scale()=scale;
    dct2.scale()=scale;
/*
   decryptor.decrypt(ct2,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of C2S: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]*25.0<<" ";
    }
    cout <<endl;
    cout <<endl;
*/
    auto t5 = chrono::high_resolution_clock::now();
    chrono::duration<double> t_c2s = t5 - t4;

    cout <<"    C2S. time = "<<t_c2s.count()<<" s. "<<endl;


    cout <<"    Modulus chain index for ct2: "<< context.get_context_data(ct2.params_id()).chain_depth()<<", ";
    cout <<"scale of ct2: "<<ct2.scale()<<" log2(scale): "<<log2(ct2.scale())<<endl;

    decryptor.decrypt(ct2,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of ct2: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;

    decryptor.decrypt(dct2,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of dct2: ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;
    cout <<endl;

    //ct2.scale()=scale;
    //return ct2;

    //ct4 = EvalLUT(ct3, lut_coeff, exp_coeff), level = 9 + 10
    cout <<"4. ct4[0] = EvalLUT(ct2, lut_coeff, exp_coeff). ct4[1] = EvalLUT(dct2, lut_coeff, exp_coeff)"<<endl;


    auto t6 = chrono::high_resolution_clock::now();
    //ct3 = exp(ct2), level = 9
    PhantomCiphertext ct3 = poly_eval_2pii(ct2,relin_keys,context,exp_coeff,secret_key);
    PhantomCiphertext dct3 = poly_eval_2pii(dct2,relin_keys,context,exp_coeff,secret_key);
    auto t7 = chrono::high_resolution_clock::now();

    //cout <<"    Modulus chain index for ct3: "<< context.get_context_data(ct3.parms_id())->chain_index()<<", ";
    //cout <<"scale of ct3: "<<ct3.scale()<<" log2(scale): "<<log2(ct3.scale())<<endl;


    //return ct3;


    auto t8 = chrono::high_resolution_clock::now();
    vector<PhantomCiphertext> ct4(2);
    Bootstrap_RangeCheck_PatersonStockmeyer(ct4[0], ct3, lut_coeff, relin_keys, context, secret_key);
    Bootstrap_RangeCheck_PatersonStockmeyer(ct4[1], dct3, lut_coeff, relin_keys, context, secret_key);

    auto t9 = chrono::high_resolution_clock::now();

    //ct4.scale()=scale;




    chrono::duration<double> t_lut1 = (t7 - t6);
    chrono::duration<double> t_lut2 = (t9 - t8);

    cout <<"    EvalLUT. time = "<<t_lut1.count()<<" s. + "<<t_lut2.count()<<" s. "<<endl;

    cout <<"    Modulus chain index for ct4: "<< context.get_context_data(ct4[0].params_id()).chain_depth()<<", ";
    cout <<"scale of ct4: "<<ct4[0].scale()<<" log2(scale): "<<log2(ct4[0].scale())<<endl;

    decryptor.decrypt(ct3,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of EvalPoly 2pii(ct2): ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;

    decryptor.decrypt(dct3,pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of EvalPoly 2pii(dct2): ";
    for (int i = 0; i < 10; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;

    decryptor.decrypt(ct4[0],pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of EvalPoly 512(ct3): ";
    for (int i = 0; i < 100; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;

    decryptor.decrypt(ct4[1],pt_output);
    encoder.decode(pt_output,dcd_output);
    cout <<"    Decrypt result of EvalPoly 512(dct3): ";
    for (int i = 0; i < 100; ++i){
        cout <<dcd_output[i]<<" ";
    }
    cout <<endl;
    cout <<endl;


    cout <<"Total time = "<<t_s2c.count()+t_mr.count()+t_c2s.count()+t_lut1.count()+t_lut2.count()<<" s. "<<endl;

    return ct4;

}



// #include <chrono>
// using namespace chrono;
// #include "Bootstrapper.h"
// #include "ckks_evaluator.h"

// using namespace std;
// using namespace seal;

// vector<Ciphertext> FuncBoot_slot_v2_2ct(const Ciphertext & ct, const Ciphertext & dct, const vector<complex<double>> & lut_coeff,
//     const vector<complex<double>> & exp_coeff, const RelinKeys &relin_keys, Bootstrapper& bootstrapper,
//     int poly_modulus_degree, int slot_count, const SEALContext& context, const SecretKey & secret_key){

//     cout <<"------------Functional bootstrapping--------------------"<<endl;


//     bootstrapper.initial_scale = ct.scale();
//     double scale = ct.scale();
//    // cout <<"    scale of input ct: "<<ct.scale()<<" log2(scale): "<<log2(ct.scale())<<endl;
//     CKKSEncoder encoder(context);
//     Evaluator evaluator(context,encoder);

//     //for test
//     Decryptor decryptor(context, secret_key);

//     Plaintext pt_output;
//     vector<complex<double>> dcd_output;

//     Ciphertext input1 = ct;
//     Ciphertext input2 = dct;

//     const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
//     double q0 = (double)modulus[0].value();
//     if(scale != q0){
//         cout <<"Incorrect scale of input ct. "<<endl;
//     }
//     /*
//     decryptor.decrypt(input1,pt_output);
//     encoder.decode(pt_output,dcd_output);

//     cout <<"Decrypt result of input1: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i].real()<<" ";
//     }
//     cout <<endl;
//     */
//     cout <<endl;


//     //cout <<poly_modulus_degree<<" "<<slot_count<<endl;




//     //input1 = input1/p
//     cout <<"0. input1 = input1/512. input2 = input2/512"<<endl;
//     Plaintext ptp;
//     encoder.encode(1/512.0, input1.parms_id(), input1.scale(),ptp);
//     evaluator.multiply_plain_inplace(input1,ptp);
//     evaluator.rescale_to_next_inplace(input1);

//     evaluator.multiply_plain_inplace(input2,ptp);
//     evaluator.rescale_to_next_inplace(input2);

//     cout <<"    Modulus chain index for input ct: "<< context.get_context_data(input1.parms_id())->chain_index()<<", ";
//     cout <<"scale of input ct: "<<input1.scale()<<" log2(scale): "<<log2(input1.scale())<<endl;

//     input1.scale()=scale;

//     decryptor.decrypt(input1,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of input1: ";
//     for (int i = 0; i < 15; ++i){
//         cout <<dcd_output[i].real()<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(input2,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of input2: ";
//     for (int i = 0; i < 15; ++i){
//         cout <<dcd_output[i].real()<<" ";
//     }
//     cout <<endl;
//     cout<<endl;

//     //ct1 = S2C(ct), level = 3
//     cout <<"1. ct1 = S2C(input1,input2). "<<endl;

//    // cout <<"    Modulus chain index for input ct: "<< context.get_context_data(input1.parms_id())->chain_index()<<", ";
//    // cout <<"scale of input ct: "<<input1.scale()<<" log2(scale): "<<log2(input1.scale())<<endl;
//     //S2C start
//     auto t0 = chrono::high_resolution_clock::now();


//     //Ciphertext input2 = ct;
//     Ciphertext ct1;

//     bootstrapper.slottocoeff_full_3_fb(ct1,input1,input2);

//     ct1.scale()=scale;

//     auto t1 = chrono::high_resolution_clock::now();
//     chrono::duration<double> t_s2c = t1 - t0;

//     cout <<"    S2C. time = "<<t_s2c.count()<<" s. "<<endl;

//     cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.parms_id())->chain_index()<<", ";
//     cout <<"scale of ct1: "<<ct1.scale()<<" log2(scale): "<<log2(ct1.scale())<<endl;

//     decryptor.decrypt(ct1,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of S2C: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<" ...... ";
//     for (int i = slot_count; i < slot_count+10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;
//     cout <<endl;

//     //mod switch to lowest level

//     while(ct1.coeff_modulus_size() != 1) evaluator.mod_switch_to_next_inplace(ct1);

//    // cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.parms_id())->chain_index()<<endl;


//     //ct2 = ct1 = ModRaise(ct1)
//     cout <<"2. ct1 = ModRaise(ct1). "<<endl;
//     auto t2 = chrono::high_resolution_clock::now();

//     //ModRaise_inplace(ct1,context);
//     bootstrapper.modraise_inplace(ct1);

//     auto t3 = chrono::high_resolution_clock::now();
//     chrono::duration<double> t_mr = t3 - t2;

//     cout <<"    ModRaise. time = "<<t_mr.count()<<" s. "<<endl;

//     cout <<"    Modulus chain index for ct1: "<< context.get_context_data(ct1.parms_id())->chain_index()<<", ";
//     cout <<"scale of ct1: "<<ct1.scale()<<" log2(scale): "<<log2(ct1.scale())<<endl;

//     decryptor.decrypt(ct1,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of ModRaise: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;
//     cout <<endl;

//     //ct3 = C2S(ct2), level = 3
//     cout <<"3. ct2 = C2S(ct1)/25.0. "<<endl;

//     //C2S start
//     auto t4 = chrono::high_resolution_clock::now();

//     Ciphertext ct2, dct2;

//     bootstrapper.coefftoslot_full_3(ct2,dct2,ct1);

//     ct2.scale()=scale;
//     dct2.scale()=scale;
// /*
//    decryptor.decrypt(ct2,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of C2S: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]*25.0<<" ";
//     }
//     cout <<endl;
//     cout <<endl;
// */
//     auto t5 = chrono::high_resolution_clock::now();
//     chrono::duration<double> t_c2s = t5 - t4;

//     cout <<"    C2S. time = "<<t_c2s.count()<<" s. "<<endl;


//      cout <<"    Modulus chain index for ct2: "<< context.get_context_data(ct2.parms_id())->chain_index()<<", ";
//     cout <<"scale of ct2: "<<ct2.scale()<<" log2(scale): "<<log2(ct2.scale())<<endl;

//     decryptor.decrypt(ct2,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of ct2: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(dct2,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of dct2: ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;
//     cout <<endl;

//     //ct2.scale()=scale;
//     //return ct2;

//     //ct4 = EvalLUT(ct3, lut_coeff, exp_coeff), level = 9 + 10
//     cout <<"4. ct4[0] = EvalLUT(ct2, lut_coeff, exp_coeff). ct4[1] = EvalLUT(dct2, lut_coeff, exp_coeff)"<<endl;


//     auto t6 = chrono::high_resolution_clock::now();

//     //ct3 = exp(ct2), level = 9
//     Ciphertext ct3 = poly_eval_2pii(ct2,relin_keys,context,exp_coeff,secret_key);
//     Ciphertext dct3 = poly_eval_2pii(dct2,relin_keys,context,exp_coeff,secret_key);

//     auto t7 = chrono::high_resolution_clock::now();

//     //cout <<"    Modulus chain index for ct3: "<< context.get_context_data(ct3.parms_id())->chain_index()<<", ";
//     //cout <<"scale of ct3: "<<ct3.scale()<<" log2(scale): "<<log2(ct3.scale())<<endl;


//     //return ct3;


//     auto t8 = chrono::high_resolution_clock::now();
//     vector<Ciphertext> ct4(2);
//     Bootstrap_RangeCheck_PatersonStockmeyer(ct4[0], ct3, lut_coeff, relin_keys, context, secret_key);
//     Bootstrap_RangeCheck_PatersonStockmeyer(ct4[1], dct3, lut_coeff, relin_keys, context, secret_key);

//     auto t9 = chrono::high_resolution_clock::now();

//     //ct4.scale()=scale;




//     chrono::duration<double> t_lut1 = (t7 - t6);
//     chrono::duration<double> t_lut2 = (t9 - t8);

//     cout <<"    EvalLUT. time = "<<t_lut1.count()<<" s. + "<<t_lut2.count()<<" s. "<<endl;

//     cout <<"    Modulus chain index for ct4: "<< context.get_context_data(ct4[0].parms_id())->chain_index()<<", ";
//     cout <<"scale of ct4: "<<ct4[0].scale()<<" log2(scale): "<<log2(ct4[0].scale())<<endl;

//     decryptor.decrypt(ct3,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of EvalPoly 2pii(ct2): ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(dct3,pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of EvalPoly 2pii(dct2): ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(ct4[0],pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of EvalPoly 512(ct3): ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;

//     decryptor.decrypt(ct4[1],pt_output);
//     encoder.decode(pt_output,dcd_output);
//     cout <<"    Decrypt result of EvalPoly 512(dct3): ";
//     for (int i = 0; i < 10; ++i){
//         cout <<dcd_output[i]<<" ";
//     }
//     cout <<endl;
//     cout <<endl;


//     cout <<"Total time = "<<t_s2c.count()+t_mr.count()+t_c2s.count()+t_lut1.count()+t_lut2.count()<<" s. "<<endl;

//     return ct4;

// }
