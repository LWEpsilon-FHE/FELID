#pragma once
#include "include.cuh"

extern PhantomCiphertext poly_eval_2pii(const PhantomCiphertext & RLWE_ct,
const PhantomRelinKey & relin_keys, PhantomContext& context, const vector<complex<double>> &f,
PhantomSecretKey & sk) {
    // cout << "[DEBUG]total_coeff_modulus_bit_count_ = " << context.get_context_data(RLWE_ct.params_id()).total_coeff_modulus_bit_count() << endl;
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder = Encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context,&phantom_encoder);

    //for test
    // Decryptor decryptor(&context, &sk);

    double scale = RLWE_ct.scale();
    // cout << "[DEBUG] scale used in PolyEval_2: "<<scale<<" log2(scale): "<<log2(scale)<<endl;

    int DegreeK = f.size()-1;
    if (DegreeK == 0){
        cout <<"Unsupport: consstant polynomial. "<<endl;
    }

    vector<PhantomCiphertext> Tx(DegreeK);
    Tx[0] = RLWE_ct;

    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication

    auto start_1 = std::chrono::high_resolution_clock::now();
    for (int i = 2; i <= DegreeK; ++i){
        //if i is power of two
        if(!(i & (i-1))){
            //cout <<i <<endl;
            // compute T_{i}(y) = 2*T_i/2(y)^2 - 1
            PhantomCiphertext square = Tx[i/2-1];
            //square = T_{i/2}*T_{i/2}
            evaluator.square_inplace(square);
            evaluator.relinearize_inplace(square, relin_keys);
            evaluator.rescale_to_next_inplace(square);
            //square = 2*square
            evaluator.add_inplace(square,square);
            double a = -1.0;
            PhantomPlaintext pta;
            encoder.encode(a,square.params_id(),square.scale(),pta);
            // cout << "[DEBUG] pta scale: "<<log2(pta.scale())<<", square scale: "<<log2(square.scale())<<endl;
            //square = square - 1
            evaluator.add_plain_inplace(square,pta);
            square.scale()=scale;
            //T_{i} = square
            Tx[i-1] = square;

        }
        else{
            //non power of 2
            if(i % 2 ==1){
                // if i is odd
                // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
                PhantomCiphertext prod = Tx[i/2-1];
                evaluator.mod_switch_to_inplace(prod,Tx[i/2].params_id());
                evaluator.multiply_inplace(prod, Tx[i/2]);
                evaluator.relinearize_inplace(prod, relin_keys);
                evaluator.rescale_to_next_inplace(prod);
                //prod = 2*prod
                evaluator.add_inplace(prod,prod);
                //prod = prod - 1
                //double a = -1.0;
                //Plaintext pta;
                //encoder.encode(a,prod.parms_id(),prod.scale(),pta);
                PhantomCiphertext tempx = RLWE_ct;
                evaluator.mod_switch_to_inplace(tempx,prod.params_id());
                //cout <<log2(tempx.scale())<<" "<<log2(prod.scale())<<endl;
                prod.scale()=scale;
                evaluator.sub_inplace(prod,tempx);
                //T_{i} = prod
                Tx[i-1] = prod;
            }
            else{
                // i is even but not power of 2
                // compute T_{2i}(y) = 2*T_i(y)^2 - 1
                PhantomCiphertext square = Tx[i/2-1];
                //square = T_{i/2}*T_{i/2}
                evaluator.square_inplace(square);
                evaluator.relinearize_inplace(square, relin_keys);
                evaluator.rescale_to_next_inplace(square);
                //square = 2*square
                evaluator.add_inplace(square,square);
                //square = square - 1
                double a = -1.0;
                PhantomPlaintext pta;
                encoder.encode(a,square.params_id(),square.scale(),pta);
                // cout << "[DEBUG] pta scale: "<<log2(pta.scale())<<", square scale: "<<log2(square.scale())<<endl;
                evaluator.add_plain_inplace(square,pta);
                square.scale()=scale;
                //T_{i} = square
                Tx[i-1] = square;
            }
        }
    }
    cudaDeviceSynchronize();
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_1 = end_1 - start_1;
    // cout << "[DEBUG] Time for computing Chebyshev polynomials up to degree k: " << diff_1.count() << " s" << endl;



    auto start_2 = std::chrono::high_resolution_clock::now();
    PhantomCiphertext res;
    for (int i = 1; i <= DegreeK; ++i){
        evaluator.mod_switch_to_inplace(Tx[i-1],Tx[DegreeK-1].params_id());
/*
        //test decrypt ct_output
        Plaintext pt_output;
        decryptor.decrypt(Tx[i-1],pt_output);
        vector<complex<double>> dcd_output;
        encoder.decode(pt_output,dcd_output);
        cout <<"Decrypt result of Tx "<<i<<": ";
        for (int i = 0; i < 15; ++i){
            cout <<dcd_output[i]<<" ";
        }
        cout <<endl;
*/
        PhantomPlaintext coeff;
        encoder.encode(f[i],Tx[i-1].params_id(),Tx[i-1].scale(),coeff);
        evaluator.multiply_plain_inplace(Tx[i-1],coeff);
        evaluator.rescale_to_next_inplace(Tx[i-1]);
        Tx[i-1].scale() = scale;
        if (i == 1){
            res = Tx[i-1];
        }
        else{
            evaluator.add_inplace(res,Tx[i-1]);
        }
    }
    cudaDeviceSynchronize();
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_2 = end_2 - start_2;
    // cout << "[DEBUG] Time for evaluating the polynomial: " << diff_2.count() << " s" << endl;

    PhantomPlaintext coeff0;
    encoder.encode(f[0],res.params_id(),res.scale(),coeff0);
    evaluator.add_plain_inplace(res,coeff0);

    evaluator.mod_switch_to_next_inplace(res);        //modified on Sep 6

    evaluator.square_inplace(res);
    evaluator.relinearize_inplace(res, relin_keys);
    evaluator.rescale_to_next_inplace(res);
    res.scale()=scale;

    evaluator.square_inplace(res);
    evaluator.relinearize_inplace(res, relin_keys);
    evaluator.rescale_to_next_inplace(res);
    res.scale()=scale;

    return res;
}



extern PhantomCiphertext poly_eval_2pii_parallel(const PhantomCiphertext & RLWE_ct,
const PhantomRelinKey & relin_keys, PhantomContext& context, const vector<complex<double>> &f,
PhantomSecretKey & sk, phantom::util::cuda_stream_wrapper& stream = *phantom::util::global_variables::default_stream) {
    // cout << "[DEBUG]total_coeff_modulus_bit_count_ = " << context.get_context_data(RLWE_ct.params_id()).total_coeff_modulus_bit_count() << endl;
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder = Encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context,&phantom_encoder);

    //for test
    Decryptor decryptor(&context, &sk);

    double scale = RLWE_ct.scale();
    // cout << "[DEBUG] scale used in PolyEval_2: "<<scale<<" log2(scale): "<<log2(scale)<<endl;

    int DegreeK = f.size()-1;
    if (DegreeK == 0){
        cout <<"Unsupport: consstant polynomial. "<<endl;
    }

    vector<PhantomCiphertext> Tx(DegreeK);
    // Tx[0] = RLWE_ct;
    Tx[0] = deep_copy_cipher(RLWE_ct, context, stream);

    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication
    // double a = -1.0;
    // PhantomPlaintext pta;
    // encoder.encode(a, scale, pta, stream);

    auto start_1 = std::chrono::high_resolution_clock::now();
    for (int i = 2; i <= DegreeK; ++i){
        //if i is power of two
        if(!(i & (i-1))){
            //cout <<i <<endl;
            // compute T_{i}(y) = 2*T_i/2(y)^2 - 1
            // PhantomCiphertext square = Tx[i/2-1];
            PhantomCiphertext square = deep_copy_cipher(Tx[i/2-1], context, stream);
            //square = T_{i/2}*T_{i/2}
            evaluator.square_inplace(square, stream);
            evaluator.relinearize_inplace(square, relin_keys, stream);
            evaluator.rescale_to_next_inplace(square, stream);
            //square = 2*square
            evaluator.add_inplace(square,square, stream);

            double a = -1.0;
            PhantomPlaintext pta;
            encoder.encode(a, square.params_id(), square.scale(), pta, stream);

            // evaluator.mod_switch_to_inplace(pta, square.params_id(), stream);
            // cout << "[DEBUG] pta scale: "<<log2(pta.scale())<<", square scale: "<<log2(square.scale())<<endl;
            //square = square - 1
            evaluator.add_plain_inplace(square,pta, stream);
            square.scale()=scale;
            //T_{i} = square
            // Tx[i-1] = square;
            Tx[i-1] = deep_copy_cipher(square, context, stream);

        }
        else{
            //non power of 2
            if(i % 2 ==1){
                // if i is odd
                // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
                // PhantomCiphertext prod = Tx[i/2-1];
                PhantomCiphertext prod = deep_copy_cipher(Tx[i/2-1], context, stream);
                evaluator.mod_switch_to_inplace(prod,Tx[i/2].params_id(), stream);
                evaluator.multiply_inplace(prod, Tx[i/2], stream);
                evaluator.relinearize_inplace(prod, relin_keys, stream);
                evaluator.rescale_to_next_inplace(prod, stream);
                //prod = 2*prod
                evaluator.add_inplace(prod,prod, stream);
                //prod = prod - 1

                // double a = -1.0;
                // PhantomPlaintext pta;
                // encoder.encode(a, prod.params_id(), prod.scale(), pta, stream);

                // PhantomCiphertext tempx = RLWE_ct;
                PhantomCiphertext tempx = deep_copy_cipher(RLWE_ct, context, stream);
                evaluator.mod_switch_to_inplace(tempx,prod.params_id(), stream);
                //cout <<log2(tempx.scale())<<" "<<log2(prod.scale())<<endl;
                prod.scale()=scale;
                evaluator.sub_inplace(prod,tempx, stream);
                //T_{i} = prod
                // Tx[i-1] = prod;
                Tx[i-1] = deep_copy_cipher(prod, context, stream);
            }
            else{
                // i is even but not power of 2
                // compute T_{2i}(y) = 2*T_i(y)^2 - 1
                PhantomCiphertext square = deep_copy_cipher(Tx[i/2-1], context, stream);
                //square = T_{i/2}*T_{i/2}
                evaluator.square_inplace(square, stream);
                evaluator.relinearize_inplace(square, relin_keys, stream);
                evaluator.rescale_to_next_inplace(square, stream);
                //square = 2*square
                evaluator.add_inplace(square,square, stream);
                //square = square - 1

                double a = -1.0;
                PhantomPlaintext pta;
                encoder.encode(a, square.params_id(), square.scale(), pta, stream);

                evaluator.mod_switch_to_inplace(pta, square.params_id(), stream);
                // cout << "[DEBUG] pta scale: "<<log2(pta.scale())<<", square scale: "<<log2(square.scale())<<endl;
                evaluator.add_plain_inplace(square,pta, stream);
                square.scale()=scale;
                //T_{i} = square
                // Tx[i-1] = square;
                Tx[i-1] = deep_copy_cipher(square, context, stream);
            }
        }
    }
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_1 = end_1 - start_1;
    // cout << "[DEBUG] Time for computing Chebyshev polynomials up to degree k: " << diff_1.count() << " s" << endl;



    auto start_2 = std::chrono::high_resolution_clock::now();
    PhantomCiphertext res;
    for (int i = 1; i <= DegreeK; ++i){
        evaluator.mod_switch_to_inplace(Tx[i-1],Tx[DegreeK-1].params_id(), stream);

        //test decrypt ct_output
        // PhantomPlaintext pt_output;
        // decryptor.decrypt(Tx[i-1],pt_output);
        // vector<complex<double>> dcd_output;
        // encoder.decode(pt_output,dcd_output);
        // cout <<"Decrypt result of Tx "<<i<<": ";
        // for (int i = 0; i < 15; ++i){
        //     cout <<dcd_output[i]<<" ";
        // }
        // cout <<endl;

        PhantomPlaintext coeff;
        encoder.encode(f[i],Tx[i-1].params_id(),Tx[i-1].scale(),coeff, stream);
        evaluator.multiply_plain_inplace(Tx[i-1],coeff, stream);
        evaluator.rescale_to_next_inplace(Tx[i-1], stream);
        Tx[i-1].scale() = scale;
        if (i == 1){
            res = Tx[i-1];
        }
        else{
            evaluator.add_inplace(res,Tx[i-1], stream);
        }
    }
    // cudaDeviceSynchronize();


    PhantomPlaintext coeff0;
    encoder.encode(f[0],res.params_id(),res.scale(),coeff0, stream);
    evaluator.add_plain_inplace(res,coeff0, stream);

    evaluator.mod_switch_to_next_inplace(res, stream);        //modified on Sep 6

    evaluator.square_inplace(res, stream);
    evaluator.relinearize_inplace(res, relin_keys, stream);
    evaluator.rescale_to_next_inplace(res, stream);
    res.scale()=scale;

    evaluator.square_inplace(res, stream);
    evaluator.relinearize_inplace(res, relin_keys, stream);
    evaluator.rescale_to_next_inplace(res, stream);
    res.scale()=scale;

    // PhantomCiphertext ret;
    // ret = deep_copy_cipher(res, context);
    // cudaStreamSynchronize(stream.get_stream());

    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_2 = end_2 - start_2;
    // cout << "[DEBUG] Time for evaluating the polynomial: " << diff_2.count() << " s" << endl;

    // cout << "total time in poly_eval_2pii_parallel: " << diff_1.count() + diff_2.count() << " s" << endl;
    cudaStreamSynchronize(stream.get_stream());
    return res;
}



// using namespace std;
// using namespace seal;

// extern Ciphertext poly_eval_2pii(const Ciphertext & RLWE_ct,
// const RelinKeys & relin_keys, const SEALContext& seal_context, const vector<complex<double>> & f,
// const SecretKey & sk){

//     CKKSEncoder encoder(seal_context);
//     Evaluator evaluator(seal_context,encoder);

//     //for test
//     Decryptor decryptor(seal_context, sk);

//     double scale = RLWE_ct.scale();

//     int DegreeK = f.size()-1;
//     if (DegreeK == 0){
//         cout <<"Unsupport: consstant polynomial. "<<endl;
//     }

//     vector<Ciphertext> Tx(DegreeK);
//     Tx[0] = RLWE_ct;

//     // Computes Chebyshev polynomials up to degree k
//     // for y: T_1(y) = y, T_2(y), ... , T_k(y)
//     // uses binary tree multiplication

//     for (int i = 2; i <= DegreeK; ++i){
//         //if i is power of two
//         if(!(i & (i-1))){
//             //cout <<i <<endl;
//             // compute T_{i}(y) = 2*T_i/2(y)^2 - 1
//             Ciphertext square = Tx[i/2-1];
//             //square = T_{i/2}*T_{i/2}
//             evaluator.square_inplace(square);
//             evaluator.relinearize_inplace(square, relin_keys);
//             evaluator.rescale_to_next_inplace(square);
//             //square = 2*square
//             evaluator.add_inplace(square,square);
//             //square = square - 1
//             double a = -1.0;
//             Plaintext pta;
//             encoder.encode(a,square.parms_id(),square.scale(),pta);
//             evaluator.add_plain_inplace(square,pta);
//             square.scale()=scale;
//             //T_{i} = square
//             Tx[i-1] = square;

//         }
//         else{
//             //non power of 2
//             if(i % 2 ==1){
//                 // if i is odd
//                 // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
//                 Ciphertext prod = Tx[i/2-1];
//                 evaluator.mod_switch_to_inplace(prod,Tx[i/2].parms_id());
//                 evaluator.multiply_inplace(prod, Tx[i/2]);
//                 evaluator.relinearize_inplace(prod, relin_keys);
//                 evaluator.rescale_to_next_inplace(prod);
//                 //prod = 2*prod
//                 evaluator.add_inplace(prod,prod);
//                 //prod = prod - 1
//                 //double a = -1.0;
//                 //Plaintext pta;
//                 //encoder.encode(a,prod.parms_id(),prod.scale(),pta);
//                 Ciphertext tempx = RLWE_ct;
//                 evaluator.mod_switch_to_inplace(tempx,prod.parms_id());
//                 //cout <<log2(tempx.scale())<<" "<<log2(prod.scale())<<endl;
//                 prod.scale()=scale;
//                 evaluator.sub_inplace(prod,tempx);
//                 //T_{i} = prod
//                 Tx[i-1] = prod;

//             }
//             else{
//                 // i is even but not power of 2
//                 // compute T_{2i}(y) = 2*T_i(y)^2 - 1
//                 Ciphertext square = Tx[i/2-1];
//                 //square = T_{i/2}*T_{i/2}
//                 evaluator.square_inplace(square);
//                 evaluator.relinearize_inplace(square, relin_keys);
//                 evaluator.rescale_to_next_inplace(square);
//                 //square = 2*square
//                 evaluator.add_inplace(square,square);
//                 //square = square - 1
//                 double a = -1.0;
//                 Plaintext pta;
//                 encoder.encode(a,square.parms_id(),square.scale(),pta);
//                 evaluator.add_plain_inplace(square,pta);
//                 square.scale()=scale;
//                 //T_{i} = square
//                 Tx[i-1] = square;
//             }
//         }
//     }



//     Ciphertext res;
//     for (int i = 1; i <= DegreeK; ++i){
//         evaluator.mod_switch_to_inplace(Tx[i-1],Tx[DegreeK-1].parms_id());
// /*
//         //test decrypt ct_output
//         Plaintext pt_output;
//         decryptor.decrypt(Tx[i-1],pt_output);
//         vector<complex<double>> dcd_output;
//         encoder.decode(pt_output,dcd_output);
//         cout <<"Decrypt result of Tx "<<i<<": ";
//         for (int i = 0; i < 15; ++i){
//             cout <<dcd_output[i]<<" ";
//         }
//         cout <<endl;
// */
//         Plaintext coeff;
//         encoder.encode(f[i],Tx[i-1].parms_id(),Tx[i-1].scale(),coeff);
//         evaluator.multiply_plain_inplace(Tx[i-1],coeff);
//         evaluator.rescale_to_next_inplace(Tx[i-1]);
//         Tx[i-1].scale() = scale;
//         if (i == 1){
//             res = Tx[i-1];
//         }
//         else{
//             evaluator.add_inplace(res,Tx[i-1]);
//         }
//     }

//     Plaintext coeff0;
//     encoder.encode(f[0],res.parms_id(),res.scale(),coeff0);
//     evaluator.add_plain_inplace(res,coeff0);

//     evaluator.mod_switch_to_next_inplace(res);        //modified on Sep 6

//     evaluator.square_inplace(res);
//     evaluator.relinearize_inplace(res, relin_keys);
//     evaluator.rescale_to_next_inplace(res);
//     res.scale()=scale;

//     evaluator.square_inplace(res);
//     evaluator.relinearize_inplace(res, relin_keys);
//     evaluator.rescale_to_next_inplace(res);
//     res.scale()=scale;

//     return res;
// }
