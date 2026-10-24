#pragma once
#include "include.cuh"

// //compute RLWE(ecd(f(x))) from RLWE(ecd(x)) and f(x)
// extern PhantomCiphertext poly_eval(const PhantomCiphertext & RLWE_ct,
// const PhantomRelinKey & relin_keys, PhantomContext& context, const vector<complex<double>> & f,
// PhantomSecretKey & sk){

//   PhantomCKKSEncoder phantom_encoder(context);
//   Encoder encoder(&context, &phantom_encoder);
//   Evaluator evaluator(&context, &phantom_encoder);

//   //for test
//   Decryptor decryptor(&context, &sk);

//   double scale = RLWE_ct.scale();


//   int DegreeK = f.size()-1;
//   if (DegreeK == 0){
//       cout <<"Unsupport: consstant polynomial. "<<endl;
//   }

//   int num_cal = 0;

//     vector<int> calculated(DegreeK, 0); // 0 is for degree 1, 1 is for degree 2
//     calculated[0] = 1;
//     vector<int> Tocalculate(DegreeK,0);

//     vector<PhantomCiphertext> output(DegreeK);

//     output[0] = RLWE_ct;


// /*
//     //skip odd degree if skip_odd = true
//     for(int i = DegreeK; i > 0; --i){
//         if (skip_odd && i % 2 == 1) { // 0 is for degree 1, 1 is for degree 2, skip all 2k+1 degree
//             calculated[i-1] = 1;
//             output[i-1] = input;
//         }
//     }
// */

//     //step 1, compute x^2, x^4, x^8,....in sequence
//     int index = 1;
//     int num_cal_1 = 1;
//     //cout <<"calculated in step 1: "<<endl;
//     while(index < DegreeK){
//         int new_index = index * 2;
//         if (new_index > DegreeK){
//             break;
//         }
//         PhantomCiphertext temp = output[index-1];
//         evaluator.square_inplace(temp);
//         evaluator.relinearize_inplace(temp, relin_keys);
//         evaluator.rescale_to_next_inplace(temp);
//         //cout <<"square end. "<<endl;
//         output[new_index-1] = temp;
//         output[new_index-1].scale()=scale;
//         calculated[new_index-1] = 1;
//         num_cal_1++;
//         //cout <<new_index<<" ";
//         index = new_index;
//     }
//     //cout <<endl;

//     //step 2, compute x^{1,2,4,...,}*x^{1,2,4,...}
//     //cout <<"calculated in step 2: "<<endl;

//     //#pragma omp parallel for

//     for(int ii = 0 ; ii < num_cal_1 ; ++ii){
//         int i = 1 <<ii;
//         //cout <<"i = "<<i<<endl;
//         for(int j = i*2 ; j < DegreeK ; j *= 2){
//             //cout <<"j = "<<j<<", i+j = "<<i+j<<endl;
//             if(i + j > DegreeK){
//                 break;
//             }
//             int new_index = i + j;
//             if (i == j){
//                 continue;
//             }
//             if (calculated[new_index-1] == 1){
//                continue;
//             }
//             else{
//                 PhantomCiphertext temp = output[i-1];
//                 PhantomCiphertext resct = output[j-1];
//                 evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                 evaluator.multiply_inplace(temp, resct);
//                 evaluator.relinearize_inplace(temp, relin_keys);
//                 evaluator.rescale_to_next_inplace(temp);
//                 output[new_index-1] = temp;
//                 output[new_index-1].scale()=scale;
//                 calculated[new_index-1] = 1;
//                 //cout <<new_index <<" ";
//             }
//         }
//     }
//     //cout <<endl;

//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         PhantomCiphertext temp = output[i-1];
//                         PhantomCiphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     //cout <<endl;
//     cout <<"R1: number of calculated ct = "<<num_cal<<endl;
// */
//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         PhantomCiphertext temp = output[i-1];
//                         PhantomCiphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     //cout <<endl;
//     cout <<"R2: number of calculated ct = "<<num_cal<<endl;
// */
//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         PhantomCiphertext temp = output[i-1];
//                         PhantomCiphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     cout <<endl;
//     cout <<"R3: number of calculated ct = "<<num_cal<<endl;
// */
//   //  cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output[DegreeK-1].parms_id())->chain_index()<<endl;

//     PhantomCiphertext res;
//   for (int i = 1; i <= DegreeK; ++i){
//     evaluator.mod_switch_to_inplace(output[i-1],output[DegreeK-1].parms_id());

//     PhantomPlaintext coeff;
//     encoder.encode(f[i],output[i-1].parms_id(),output[i-1].scale(),coeff);
//     evaluator.multiply_plain_inplace(output[i-1],coeff);
//     evaluator.rescale_to_next_inplace(output[i-1]);
//     output[i-1].scale() = scale;
//     if (i == 1){
//       res = output[i-1];
//     }
//     else{
//       evaluator.add_inplace(res,output[i-1]);
//     }
//   }

//   PhantomPlaintext coeff0;
//   encoder.encode(f[0],res.parms_id(),res.scale(),coeff0);
//   evaluator.add_plain_inplace(res,coeff0);

//   return res;


// }

// //compute RLWE(ecd(f(x))) from RLWE(ecd(x)) and f(x)
// extern PhantomCiphertext poly_eval_small(const Ciphertext & RLWE_ct,
// const RelinKeys & relin_keys, const SEALContext& seal_context, const vector<complex<double>> & f,
// const SecretKey & sk){

//   CKKSEncoder encoder(seal_context);
//   Evaluator evaluator(seal_context,encoder);

//   //for test
//   Decryptor decryptor(seal_context, sk);

//   double scale = RLWE_ct.scale();


//   int f_degree = f.size()-1;
//   //cout <<f_degree<<endl;

//   vector<Ciphertext> x_n(f_degree+1);
//   x_n[1] = RLWE_ct;

//   evaluator.square(x_n[1],x_n[2]);
//   evaluator.relinearize_inplace(x_n[2],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[2]);

//   evaluator.square(x_n[2],x_n[4]);
//   evaluator.relinearize_inplace(x_n[4],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[4]);

//   evaluator.square(x_n[4],x_n[8]);
//   evaluator.relinearize_inplace(x_n[8],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[8]);

//   //compute x^3,5,9
//   for (int i = 2; i < f_degree; i *= 2){
//     //cout<<i+1<<" ";
//     evaluator.mod_switch_to_inplace(x_n[1],x_n[i].parms_id());
//     evaluator.multiply(x_n[1],x_n[i],x_n[i+1]);
//     evaluator.relinearize_inplace(x_n[i+1],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+1]);
//   }

//   //compute x^6,10
//   for (int i = 4; i < f_degree; i *= 2){
//     //cout<<i+2<<" ";
//     evaluator.mod_switch_to_inplace(x_n[2],x_n[i].parms_id());
//     evaluator.multiply(x_n[2],x_n[i],x_n[i+2]);
//     evaluator.relinearize_inplace(x_n[i+2],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+2]);
//   }


//   //compute x^7,11
//   for (int i = 4; i < f_degree; i *= 2){
//     //cout<<i+3<<" ";
//     evaluator.mod_switch_to_inplace(x_n[3],x_n[i].parms_id());
//     evaluator.multiply(x_n[3],x_n[i],x_n[i+3]);
//     evaluator.relinearize_inplace(x_n[i+3],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+3]);
//   }

//   //compute x^12
//   for (int i = 8; i < f_degree; i *= 2){

//     evaluator.mod_switch_to_inplace(x_n[4],x_n[i].parms_id());
//     evaluator.multiply(x_n[4],x_n[i],x_n[i+4]);
//     evaluator.relinearize_inplace(x_n[i+4],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+4]);
//   }

//   //compute x^13
//   for (int i = 8; i < f_degree; i *= 2){
//     evaluator.mod_switch_to_inplace(x_n[5],x_n[i].parms_id());
//     evaluator.multiply(x_n[5],x_n[i],x_n[i+5]);
//     evaluator.relinearize_inplace(x_n[i+5],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+5]);
//   }

//   Ciphertext res;
//   for (int i = 1; i <= f_degree; ++i){
//     evaluator.mod_switch_to_inplace(x_n[i],x_n[f_degree].parms_id());

//     Plaintext coeff;
//     encoder.encode(f[i],x_n[i].parms_id(),x_n[i].scale(),coeff);
//     evaluator.multiply_plain_inplace(x_n[i],coeff);
//     evaluator.rescale_to_next_inplace(x_n[i]);
//     x_n[i].scale() = scale;
//     if (i == 1){
//       res = x_n[i];
//     }
//     else{
//       evaluator.add_inplace(res,x_n[i]);
//     }
//   }

//   //cout <<"sum. "<<endl;
//   //compute res += ecd(a[24])
//   Plaintext coeff0;
//   encoder.encode(f[0],res.parms_id(),res.scale(),coeff0);
//   evaluator.add_plain_inplace(res,coeff0);
//   //ptsum += coeff_high_to_low[24];
//   //cout <<"pt result = "<<ptsum<<endl;

//   return res;
// }



inline void calUptoDegreeK(vector<PhantomCiphertext>& x_n, const PhantomCiphertext& input, const int DegreeK, const PhantomRelinKey &relin_keys,
                           PhantomContext& context, const bool skip_odd=false) {
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    double scale = input.scale();

    if (DegreeK < 16){
        cout <<"Unsupported degree. "<<endl;
    }

    //compute x^2,4,8,16,32
    //x_n[i-1] -> x^i
    x_n[0] = input;
    evaluator.square(x_n[0],x_n[1]);
    evaluator.relinearize_inplace(x_n[1],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[1]);
    x_n[1].scale()=scale;

    evaluator.square(x_n[1],x_n[3]);
    evaluator.relinearize_inplace(x_n[3],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[3]);
    x_n[3].scale()=scale;

    evaluator.square(x_n[3],x_n[7]);
    evaluator.relinearize_inplace(x_n[7],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[7]);
    x_n[7].scale()=scale;

    evaluator.square(x_n[7],x_n[15]);
    evaluator.relinearize_inplace(x_n[15],relin_keys);
    evaluator.rescale_to_next_inplace(x_n[15]);
    x_n[15].scale()=scale;

    if (DegreeK > 16){
        evaluator.square(x_n[15],x_n[31]);
        evaluator.relinearize_inplace(x_n[31],relin_keys);
        evaluator.rescale_to_next_inplace(x_n[31]);
        x_n[31].scale()=scale;
    }


    //compute x^3,5,9,17
    //#pragma omp parallel for

    for (int i = 2; i < DegreeK; i = i * 2){
        //cout<<i+1<<" ";
        evaluator.mod_switch_to_inplace(x_n[0],x_n[i-1].params_id());
        evaluator.multiply(x_n[0],x_n[i-1],x_n[i]);
        evaluator.relinearize_inplace(x_n[i],relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i]);
        x_n[i].scale()=scale;
    }

    //compute x^6,10,18
    //#pragma omp parallel for

    for (int i = 4; i < DegreeK; i = i * 2){
        //cout<<i+2<<" ";
        evaluator.mod_switch_to_inplace(x_n[1],x_n[i-1].params_id());
        evaluator.multiply(x_n[1],x_n[i-1],x_n[i+1]);
        evaluator.relinearize_inplace(x_n[i+1],relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i+1]);
        x_n[i+1].scale()=scale;
    }

    //compute x^7,11,19
    //#pragma omp parallel for

    for (int i = 4; i < DegreeK; i = i * 2){
        //cout<<i+3<<" ";
        evaluator.mod_switch_to_inplace(x_n[2],x_n[i-1].params_id());
        evaluator.multiply(x_n[2],x_n[i-1],x_n[i+2]);
        evaluator.relinearize_inplace(x_n[i+2],relin_keys);
        evaluator.rescale_to_next_inplace(x_n[i+2]);
        x_n[i+2].scale()=scale;
    }

    //compute x^12,20, 13,21, 14,22, 15,23

    //#pragma omp parallel for

    for(int j = 4 ; j < 8 ; ++j){
        for (int i = 8; i < DegreeK; i *= 2){
            evaluator.mod_switch_to_inplace(x_n[j-1],x_n[i-1].params_id());
            evaluator.multiply(x_n[j-1],x_n[i-1],x_n[i+j-1]);
            evaluator.relinearize_inplace(x_n[i+j-1],relin_keys);
            evaluator.rescale_to_next_inplace(x_n[i+j-1]);
            x_n[i+j-1].scale()=scale;
        }
    }

    //compute x^24, 25, 26, 27, 28, 29, 30, 31
    if (DegreeK > 16){
        //#pragma omp parallel for

        for (int j = 8; j < 16; ++j){
            evaluator.mod_switch_to_inplace(x_n[j-1],x_n[15].params_id());
            evaluator.multiply(x_n[j-1],x_n[15],x_n[16+j-1]);
            evaluator.relinearize_inplace(x_n[16+j-1],relin_keys);
            evaluator.rescale_to_next_inplace(x_n[16+j-1]);
            x_n[16+j-1].scale()=scale;
        }
    }




}

inline void calUptoDegreeK_parallel(vector<PhantomCiphertext>& x_n, const PhantomCiphertext& input, const int DegreeK, const PhantomRelinKey &relin_keys,
                           PhantomContext& context, const bool skip_odd=false, phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream) {
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);
    double scale = input.scale();

    if (DegreeK < 16){
        cout <<"Unsupported degree. "<<endl;
    }

    //compute x^2,4,8,16,32
    //x_n[i-1] -> x^i
    // x_n[0] = input;
    x_n[0] = deep_copy_cipher(input, context, stream);
    evaluator.square(x_n[0],x_n[1], stream);
    evaluator.relinearize_inplace(x_n[1],relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[1], stream);
    x_n[1].scale()=scale;

    evaluator.square(x_n[1],x_n[3], stream);
    evaluator.relinearize_inplace(x_n[3],relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[3], stream);
    x_n[3].scale()=scale;

    evaluator.square(x_n[3],x_n[7], stream);
    evaluator.relinearize_inplace(x_n[7],relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[7], stream);
    x_n[7].scale()=scale;

    evaluator.square(x_n[7],x_n[15], stream);
    evaluator.relinearize_inplace(x_n[15],relin_keys, stream);
    evaluator.rescale_to_next_inplace(x_n[15], stream);
    x_n[15].scale()=scale;

    if (DegreeK > 16){
        evaluator.square(x_n[15],x_n[31], stream);
        evaluator.relinearize_inplace(x_n[31],relin_keys, stream);
        evaluator.rescale_to_next_inplace(x_n[31], stream);
        x_n[31].scale()=scale;
    }


    //compute x^3,5,9,17
    //#pragma omp parallel for

    for (int i = 2; i < DegreeK; i = i * 2){
        //cout<<i+1<<" ";
        evaluator.mod_switch_to_inplace(x_n[0],x_n[i-1].params_id(), stream);
        evaluator.multiply(x_n[0],x_n[i-1],x_n[i], stream);
        evaluator.relinearize_inplace(x_n[i],relin_keys, stream);
        evaluator.rescale_to_next_inplace(x_n[i], stream);
        x_n[i].scale()=scale;
    }

    //compute x^6,10,18
    //#pragma omp parallel for

    for (int i = 4; i < DegreeK; i = i * 2){
        //cout<<i+2<<" ";
        evaluator.mod_switch_to_inplace(x_n[1],x_n[i-1].params_id(), stream);
        evaluator.multiply(x_n[1],x_n[i-1],x_n[i+1], stream);
        evaluator.relinearize_inplace(x_n[i+1],relin_keys, stream);
        evaluator.rescale_to_next_inplace(x_n[i+1], stream);
        x_n[i+1].scale()=scale;
    }

    //compute x^7,11,19
    //#pragma omp parallel for

    for (int i = 4; i < DegreeK; i = i * 2){
        //cout<<i+3<<" ";
        evaluator.mod_switch_to_inplace(x_n[2],x_n[i-1].params_id(), stream);
        evaluator.multiply(x_n[2],x_n[i-1],x_n[i+2], stream);
        evaluator.relinearize_inplace(x_n[i+2],relin_keys, stream);
        evaluator.rescale_to_next_inplace(x_n[i+2], stream);
        x_n[i+2].scale()=scale;
    }

    //compute x^12,20, 13,21, 14,22, 15,23

    //#pragma omp parallel for

    for(int j = 4 ; j < 8 ; ++j){
        for (int i = 8; i < DegreeK; i *= 2){
            evaluator.mod_switch_to_inplace(x_n[j-1],x_n[i-1].params_id(), stream);
            evaluator.multiply(x_n[j-1],x_n[i-1],x_n[i+j-1], stream);
            evaluator.relinearize_inplace(x_n[i+j-1],relin_keys, stream);
            evaluator.rescale_to_next_inplace(x_n[i+j-1], stream);
            x_n[i+j-1].scale()=scale;
        }
    }

    //compute x^24, 25, 26, 27, 28, 29, 30, 31
    if (DegreeK > 16){
        //#pragma omp parallel for

        for (int j = 8; j < 16; ++j){
            evaluator.mod_switch_to_inplace(x_n[j-1],x_n[15].params_id(), stream);
            evaluator.multiply(x_n[j-1],x_n[15],x_n[16+j-1], stream);
            evaluator.relinearize_inplace(x_n[16+j-1],relin_keys, stream);
            evaluator.rescale_to_next_inplace(x_n[16+j-1], stream);
            x_n[16+j-1].scale()=scale;
        }
    }




}


void Bootstrap_RangeCheck_PatersonStockmeyer(PhantomCiphertext& ciphertext, const PhantomCiphertext& input, const vector<complex<double>>& rangeCheckIndices,
                                            const PhantomRelinKey &relin_keys, PhantomContext& context, PhantomSecretKey& secret_key,
                                              const int firstDegree=32, const int secondDegree=16) {
    //MemoryPoolHandle my_pool_larger = MemoryPoolHandle::New(true);
    //auto old_prof_larger = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool_larger)));

    Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);

    double scale = input.scale();
    // cout <<log2(scale)<<endl;

    vector<PhantomCiphertext> kCTs(firstDegree);

    calUptoDegreeK(kCTs, input, firstDegree, relin_keys, context);

    for (int j = 0; j < (int) kCTs.size(); j++) {
      evaluator.mod_switch_to_inplace(kCTs[j], kCTs[kCTs.size()-1].params_id());
    }

    // cout <<"Modulus chain index for kCTs: "<< context.get_context_data(kCTs[firstDegree-1].params_id()).chain_depth()<<endl;
    // cout <<"scale = "<<log2(kCTs[firstDegree-1].scale())<<endl;

    //test for

    vector<PhantomCiphertext> kToMCTs(secondDegree);

    calUptoDegreeK(kToMCTs, kCTs[kCTs.size()-1], secondDegree, relin_keys, context);

    for (int j = 0; j < (int) kToMCTs.size(); j++) {
      evaluator.mod_switch_to_inplace(kToMCTs[j], kToMCTs[kToMCTs.size()-1].params_id());
    }

    // cout <<"Modulus chain index for kToMCTs: "<< context.get_context_data(kToMCTs[secondDegree-1].params_id()).chain_depth()<<endl;
    // cout <<"scale = "<<log2(kToMCTs[secondDegree-1].scale())<<endl;

    // for (int j = 0; j < (int) kCTs.size(); j++) {
    //   evaluator.mod_switch_to_inplace(kCTs[j], kToMCTs[kToMCTs.size()-1].params_id());
    // }

    // cout <<"Modulus chain index for the result: "<< context.get_context_data(kCTs[firstDegree-1].params_id()).chain_depth()<<endl;

    PhantomCiphertext temp_relin;
    vector<PhantomCiphertext> levelSum(secondDegree);

    //#pragma omp parallel for

    for(int i = 0; i < secondDegree; i++) {
        //cout <<i <<endl;
      //  Ciphertext levelSum;
        bool flag = false;


        for(int j = 0; j < firstDegree; j++) {
            //cout <<j <<"- ";

                PhantomPlaintext plainInd;
                if(i*firstDegree+j+1 < 512) encoder.encode(rangeCheckIndices[i*firstDegree+j+1], kCTs[j].params_id(), kCTs[j].scale(), plainInd);
                else encoder.encode(0, kCTs[j].params_id(), kCTs[j].scale(), plainInd);

                if (!flag) {
                    evaluator.multiply_plain(kCTs[j], plainInd, levelSum[i]);
                    evaluator.rescale_to_next_inplace(levelSum[i]);

                    flag = true;
                }
                else if(i*firstDegree+j+1 < 512) {
                    PhantomCiphertext tmp;
                    evaluator.multiply_plain(kCTs[j], plainInd, tmp);
                    evaluator.rescale_to_next_inplace(tmp);
                    levelSum[i].scale() = scale;
                    tmp.scale()=scale;
                //    cout <<log2(levelSum[i].scale())<<" "<<log2(tmp.scale())<<endl;
                    evaluator.add_inplace(levelSum[i], tmp);
                }
                else;
        }



        // cout <<"Modulus chain index for levelSum[i]: "<< context.get_context_data(levelSum[i].params_id()).chain_depth()<<endl;

        evaluator.mod_switch_to_inplace(levelSum[i], kToMCTs[i].params_id());
      }

    for (int i = 0; i < secondDegree; ++i){
        //cout <<i <<endl;
        if(i == 0) {
            ciphertext = levelSum[0];
        }
        else if (i == 1) { // initialize for temp_relin
            evaluator.multiply(levelSum[i], kToMCTs[i - 1], temp_relin);
            evaluator.relinearize_inplace(temp_relin, relin_keys);
            evaluator.rescale_to_next_inplace(temp_relin);
        }
        else {
            evaluator.multiply_inplace(levelSum[i], kToMCTs[i - 1]);
            evaluator.relinearize_inplace(levelSum[i], relin_keys);
            evaluator.rescale_to_next_inplace(levelSum[i]);
            // cout <<log2(levelSum[i].scale())<<" "<<log2(temp_relin.scale())<<endl;
            evaluator.add_inplace(temp_relin, levelSum[i]);
        }

    }

    evaluator.mod_switch_to_inplace(ciphertext, temp_relin.params_id());
    // cout <<"Modulus chain index for ciphertext: "<< context.get_context_data(ciphertext.params_id()).chain_depth()<<endl;
    // cout <<log2(ciphertext.scale())<<" "<<log2(temp_relin.scale())<<endl;
    ciphertext.scale()=scale;
    temp_relin.scale()=scale;
    evaluator.add_inplace(ciphertext, temp_relin);

    PhantomPlaintext coeff0;
    encoder.encode(rangeCheckIndices[0],ciphertext.params_id(),ciphertext.scale(),coeff0);
    evaluator.add_plain_inplace(ciphertext,coeff0);





}


void Bootstrap_RangeCheck_PatersonStockmeyer_parallel(PhantomCiphertext& ciphertext, const PhantomCiphertext& input, const vector<complex<double>>& rangeCheckIndices,
                                            const PhantomRelinKey &relin_keys, PhantomContext& context, PhantomSecretKey& secret_key, phantom::util::cuda_stream_wrapper &stream = *phantom::util::global_variables::default_stream,
                                              const int firstDegree=32, const int secondDegree=16) {
    //MemoryPoolHandle my_pool_larger = MemoryPoolHandle::New(true);
    //auto old_prof_larger = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool_larger)));

    // Decryptor decryptor(&context, &secret_key);
    PhantomCKKSEncoder phantom_encoder(context);
    Encoder encoder(&context, &phantom_encoder);
    Evaluator evaluator(&context, &phantom_encoder);

    double scale = input.scale();
    // cout <<log2(scale)<<endl;

    vector<PhantomCiphertext> kCTs(firstDegree);

    calUptoDegreeK_parallel(kCTs, input, firstDegree, relin_keys, context, false, stream);

    for (int j = 0; j < (int) kCTs.size(); j++) {
      evaluator.mod_switch_to_inplace(kCTs[j], kCTs[kCTs.size()-1].params_id(), stream);
    }

    // cout <<"Modulus chain index for kCTs: "<< context.get_context_data(kCTs[firstDegree-1].params_id()).chain_depth()<<endl;
    // cout <<"scale = "<<log2(kCTs[firstDegree-1].scale())<<endl;

    //test for

    vector<PhantomCiphertext> kToMCTs(secondDegree);

    calUptoDegreeK_parallel(kToMCTs, kCTs[kCTs.size()-1], secondDegree, relin_keys, context, false, stream);

    for (int j = 0; j < (int) kToMCTs.size(); j++) {
      evaluator.mod_switch_to_inplace(kToMCTs[j], kToMCTs[kToMCTs.size()-1].params_id(), stream);
    }

    // cout <<"Modulus chain index for kToMCTs: "<< context.get_context_data(kToMCTs[secondDegree-1].params_id()).chain_depth()<<endl;
    // cout <<"scale = "<<log2(kToMCTs[secondDegree-1].scale())<<endl;

    // for (int j = 0; j < (int) kCTs.size(); j++) {
    //   evaluator.mod_switch_to_inplace(kCTs[j], kToMCTs[kToMCTs.size()-1].params_id());
    // }

    // cout <<"Modulus chain index for the result: "<< context.get_context_data(kCTs[firstDegree-1].params_id()).chain_depth()<<endl;

    PhantomCiphertext temp_relin;
    vector<PhantomCiphertext> levelSum(secondDegree);

    //#pragma omp parallel for

    for(int i = 0; i < secondDegree; i++) {
        //cout <<i <<endl;
      //  Ciphertext levelSum;
        bool flag = false;


        for(int j = 0; j < firstDegree; j++) {
            //cout <<j <<"- ";

                PhantomPlaintext plainInd;
                if(i*firstDegree+j+1 < 512) encoder.encode(rangeCheckIndices[i*firstDegree+j+1], kCTs[j].params_id(), kCTs[j].scale(), plainInd, stream);
                else encoder.encode(0, kCTs[j].params_id(), kCTs[j].scale(), plainInd, stream);

                if (!flag) {
                    evaluator.multiply_plain(kCTs[j], plainInd, levelSum[i], stream);
                    evaluator.rescale_to_next_inplace(levelSum[i], stream);

                    flag = true;
                }
                else if(i*firstDegree+j+1 < 512) {
                    PhantomCiphertext tmp;
                    evaluator.multiply_plain(kCTs[j], plainInd, tmp, stream);
                    evaluator.rescale_to_next_inplace(tmp, stream);
                    levelSum[i].scale() = scale;
                    tmp.scale()=scale;
                //    cout <<log2(levelSum[i].scale())<<" "<<log2(tmp.scale())<<endl;
                    evaluator.add_inplace(levelSum[i], tmp, stream);
                }
                else;
        }



        // cout <<"Modulus chain index for levelSum[i]: "<< context.get_context_data(levelSum[i].params_id()).chain_depth()<<endl;

        evaluator.mod_switch_to_inplace(levelSum[i], kToMCTs[i].params_id(), stream);
      }

    for (int i = 0; i < secondDegree; ++i){
        //cout <<i <<endl;
        if(i == 0) {
            // ciphertext = levelSum[0];
            ciphertext = deep_copy_cipher(levelSum[0], context, stream);
        }
        else if (i == 1) { // initialize for temp_relin
            evaluator.multiply(levelSum[i], kToMCTs[i - 1], temp_relin, stream);
            evaluator.relinearize_inplace(temp_relin, relin_keys, stream);
            evaluator.rescale_to_next_inplace(temp_relin, stream);
        }
        else {
            evaluator.multiply_inplace(levelSum[i], kToMCTs[i - 1], stream);
            evaluator.relinearize_inplace(levelSum[i], relin_keys, stream);
            evaluator.rescale_to_next_inplace(levelSum[i], stream);
            // cout <<log2(levelSum[i].scale())<<" "<<log2(temp_relin.scale())<<endl;
            evaluator.add_inplace(temp_relin, levelSum[i], stream);
        }

    }

    evaluator.mod_switch_to_inplace(ciphertext, temp_relin.params_id(), stream);
    // cout <<"Modulus chain index for ciphertext: "<< context.get_context_data(ciphertext.params_id()).chain_depth()<<endl;
    // cout <<log2(ciphertext.scale())<<" "<<log2(temp_relin.scale())<<endl;
    ciphertext.scale()=scale;
    temp_relin.scale()=scale;
    evaluator.add_inplace(ciphertext, temp_relin, stream);

    PhantomPlaintext coeff0;
    encoder.encode(rangeCheckIndices[0],ciphertext.params_id(),ciphertext.scale(),coeff0, stream);
    evaluator.add_plain_inplace(ciphertext,coeff0, stream);



    cudaStreamSynchronize(stream.get_stream());
}









// using namespace std;
// using namespace seal;

// //compute RLWE(ecd(f(x))) from RLWE(ecd(x)) and f(x)
// extern Ciphertext poly_eval(const Ciphertext & RLWE_ct,
// const RelinKeys & relin_keys, const SEALContext& seal_context, const vector<complex<double>> & f,
// const SecretKey & sk){

//   CKKSEncoder encoder(seal_context);
//   Evaluator evaluator(seal_context,encoder);

//   //for test
//   Decryptor decryptor(seal_context, sk);

//   double scale = RLWE_ct.scale();


//   int DegreeK = f.size()-1;
//   if (DegreeK == 0){
//       cout <<"Unsupport: consstant polynomial. "<<endl;
//   }

//   int num_cal = 0;

//     vector<int> calculated(DegreeK, 0); // 0 is for degree 1, 1 is for degree 2
//     calculated[0] = 1;
//     vector<int> Tocalculate(DegreeK,0);

//     vector<Ciphertext> output(DegreeK);

//     output[0] = RLWE_ct;


// /*
//     //skip odd degree if skip_odd = true
//     for(int i = DegreeK; i > 0; --i){
//         if (skip_odd && i % 2 == 1) { // 0 is for degree 1, 1 is for degree 2, skip all 2k+1 degree
//             calculated[i-1] = 1;
//             output[i-1] = input;
//         }
//     }
// */

//     //step 1, compute x^2, x^4, x^8,....in sequence
//     int index = 1;
//     int num_cal_1 = 1;
//     //cout <<"calculated in step 1: "<<endl;
//     while(index < DegreeK){
//         int new_index = index * 2;
//         if (new_index > DegreeK){
//             break;
//         }
//         Ciphertext temp = output[index-1];
//         evaluator.square_inplace(temp);
//         evaluator.relinearize_inplace(temp, relin_keys);
//         evaluator.rescale_to_next_inplace(temp);
//         //cout <<"square end. "<<endl;
//         output[new_index-1] = temp;
//         output[new_index-1].scale()=scale;
//         calculated[new_index-1] = 1;
//         num_cal_1++;
//         //cout <<new_index<<" ";
//         index = new_index;
//     }
//     //cout <<endl;

//     //step 2, compute x^{1,2,4,...,}*x^{1,2,4,...}
//     //cout <<"calculated in step 2: "<<endl;

//     //#pragma omp parallel for

//     for(int ii = 0 ; ii < num_cal_1 ; ++ii){
//         int i = 1 <<ii;
//         //cout <<"i = "<<i<<endl;
//         for(int j = i*2 ; j < DegreeK ; j *= 2){
//             //cout <<"j = "<<j<<", i+j = "<<i+j<<endl;
//             if(i + j > DegreeK){
//                 break;
//             }
//             int new_index = i + j;
//             if (i == j){
//                 continue;
//             }
//             if (calculated[new_index-1] == 1){
//                continue;
//             }
//             else{
//                 Ciphertext temp = output[i-1];
//                 Ciphertext resct = output[j-1];
//                 evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                 evaluator.multiply_inplace(temp, resct);
//                 evaluator.relinearize_inplace(temp, relin_keys);
//                 evaluator.rescale_to_next_inplace(temp);
//                 output[new_index-1] = temp;
//                 output[new_index-1].scale()=scale;
//                 calculated[new_index-1] = 1;
//                 //cout <<new_index <<" ";
//             }
//         }
//     }
//     //cout <<endl;

//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         Ciphertext temp = output[i-1];
//                         Ciphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     //cout <<endl;
//     cout <<"R1: number of calculated ct = "<<num_cal<<endl;
// */
//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         Ciphertext temp = output[i-1];
//                         Ciphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     //cout <<endl;
//     cout <<"R2: number of calculated ct = "<<num_cal<<endl;
// */
//     for (int i = 1; i < DegreeK; ++i){
//         if(calculated[i-1] == 1){
//             for (int j = i ; j < DegreeK ; ++j){
//                 if (calculated[j-1] == 1){
//                     int new_index = i+j;
//                     if (new_index > DegreeK){
//                         break;
//                     }
//                     else if(calculated[new_index-1] == 0 && Tocalculate[new_index-1] == 0){
//                         //cout <<i <<" "<<j<<" "<<new_index<<endl;
//                         Ciphertext temp = output[i-1];
//                         Ciphertext resct = output[j-1];
//                         evaluator.mod_switch_to_inplace(temp,resct.parms_id());
//                         evaluator.multiply_inplace(temp, resct);
//                         evaluator.relinearize_inplace(temp, relin_keys);
//                         evaluator.rescale_to_next_inplace(temp);
//                         output[new_index-1] = temp;
//                         output[new_index-1].scale()=scale;
//                         Tocalculate[new_index-1] = 1;

//                     }
//                 }
//             }
//         }
//     }

//     for (int i = 0; i < DegreeK; ++i){
//         calculated[i] += Tocalculate[i];
//         Tocalculate[i] = 0;
//         if (calculated[i] >= 2){
//             cout <<"Wrong, i = "<<i<<endl;
//         }
//     }
// /*
//     num_cal = 0;
//     for (int i = 0; i < DegreeK; ++i){
//         if(calculated[i] == 1){
//             //cout <<i+1 <<" ";
//             num_cal ++;
//         }
//     }
//     cout <<endl;
//     cout <<"R3: number of calculated ct = "<<num_cal<<endl;
// */
//   //  cout <<"Modulus chain index for the result: "<< seal_context.get_context_data(output[DegreeK-1].parms_id())->chain_index()<<endl;

//     Ciphertext res;
//   for (int i = 1; i <= DegreeK; ++i){
//     evaluator.mod_switch_to_inplace(output[i-1],output[DegreeK-1].parms_id());

//     Plaintext coeff;
//     encoder.encode(f[i],output[i-1].parms_id(),output[i-1].scale(),coeff);
//     evaluator.multiply_plain_inplace(output[i-1],coeff);
//     evaluator.rescale_to_next_inplace(output[i-1]);
//     output[i-1].scale() = scale;
//     if (i == 1){
//       res = output[i-1];
//     }
//     else{
//       evaluator.add_inplace(res,output[i-1]);
//     }
//   }

//   Plaintext coeff0;
//   encoder.encode(f[0],res.parms_id(),res.scale(),coeff0);
//   evaluator.add_plain_inplace(res,coeff0);

//   return res;


// }

// //compute RLWE(ecd(f(x))) from RLWE(ecd(x)) and f(x)
// extern Ciphertext poly_eval_small(const Ciphertext & RLWE_ct,
// const RelinKeys & relin_keys, const SEALContext& seal_context, const vector<complex<double>> & f,
// const SecretKey & sk){

//   CKKSEncoder encoder(seal_context);
//   Evaluator evaluator(seal_context,encoder);

//   //for test
//   Decryptor decryptor(seal_context, sk);

//   double scale = RLWE_ct.scale();


//   int f_degree = f.size()-1;
//   //cout <<f_degree<<endl;

//   vector<Ciphertext> x_n(f_degree+1);
//   x_n[1] = RLWE_ct;

//   evaluator.square(x_n[1],x_n[2]);
//   evaluator.relinearize_inplace(x_n[2],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[2]);

//   evaluator.square(x_n[2],x_n[4]);
//   evaluator.relinearize_inplace(x_n[4],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[4]);

//   evaluator.square(x_n[4],x_n[8]);
//   evaluator.relinearize_inplace(x_n[8],relin_keys);
//   evaluator.rescale_to_next_inplace(x_n[8]);

//   //compute x^3,5,9
//   for (int i = 2; i < f_degree; i *= 2){
//     //cout<<i+1<<" ";
//     evaluator.mod_switch_to_inplace(x_n[1],x_n[i].parms_id());
//     evaluator.multiply(x_n[1],x_n[i],x_n[i+1]);
//     evaluator.relinearize_inplace(x_n[i+1],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+1]);
//   }

//   //compute x^6,10
//   for (int i = 4; i < f_degree; i *= 2){
//     //cout<<i+2<<" ";
//     evaluator.mod_switch_to_inplace(x_n[2],x_n[i].parms_id());
//     evaluator.multiply(x_n[2],x_n[i],x_n[i+2]);
//     evaluator.relinearize_inplace(x_n[i+2],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+2]);
//   }


//   //compute x^7,11
//   for (int i = 4; i < f_degree; i *= 2){
//     //cout<<i+3<<" ";
//     evaluator.mod_switch_to_inplace(x_n[3],x_n[i].parms_id());
//     evaluator.multiply(x_n[3],x_n[i],x_n[i+3]);
//     evaluator.relinearize_inplace(x_n[i+3],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+3]);
//   }

//   //compute x^12
//   for (int i = 8; i < f_degree; i *= 2){

//     evaluator.mod_switch_to_inplace(x_n[4],x_n[i].parms_id());
//     evaluator.multiply(x_n[4],x_n[i],x_n[i+4]);
//     evaluator.relinearize_inplace(x_n[i+4],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+4]);
//   }

//   //compute x^13
//   for (int i = 8; i < f_degree; i *= 2){
//     evaluator.mod_switch_to_inplace(x_n[5],x_n[i].parms_id());
//     evaluator.multiply(x_n[5],x_n[i],x_n[i+5]);
//     evaluator.relinearize_inplace(x_n[i+5],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[i+5]);
//   }

//   Ciphertext res;
//   for (int i = 1; i <= f_degree; ++i){
//     evaluator.mod_switch_to_inplace(x_n[i],x_n[f_degree].parms_id());

//     Plaintext coeff;
//     encoder.encode(f[i],x_n[i].parms_id(),x_n[i].scale(),coeff);
//     evaluator.multiply_plain_inplace(x_n[i],coeff);
//     evaluator.rescale_to_next_inplace(x_n[i]);
//     x_n[i].scale() = scale;
//     if (i == 1){
//       res = x_n[i];
//     }
//     else{
//       evaluator.add_inplace(res,x_n[i]);
//     }
//   }

//   //cout <<"sum. "<<endl;
//   //compute res += ecd(a[24])
//   Plaintext coeff0;
//   encoder.encode(f[0],res.parms_id(),res.scale(),coeff0);
//   evaluator.add_plain_inplace(res,coeff0);
//   //ptsum += coeff_high_to_low[24];
//   //cout <<"pt result = "<<ptsum<<endl;

//   return res;
// }



// inline void calUptoDegreeK(vector<Ciphertext>& x_n, const Ciphertext& input, const int DegreeK, const RelinKeys &relin_keys,
//                            const SEALContext& context, const bool skip_odd=false) {
//     CKKSEncoder encoder(context);
//     Evaluator evaluator(context,encoder);
//     double scale = input.scale();

//     if (DegreeK < 16){
//         cout <<"Unsupported degree. "<<endl;
//     }

//     //compute x^2,4,8,16,32
//     //x_n[i-1] -> x^i
//     x_n[0] = input;
//     evaluator.square(x_n[0],x_n[1]);
//     evaluator.relinearize_inplace(x_n[1],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[1]);
//     x_n[1].scale()=scale;

//     evaluator.square(x_n[1],x_n[3]);
//     evaluator.relinearize_inplace(x_n[3],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[3]);
//     x_n[3].scale()=scale;

//     evaluator.square(x_n[3],x_n[7]);
//     evaluator.relinearize_inplace(x_n[7],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[7]);
//     x_n[7].scale()=scale;

//     evaluator.square(x_n[7],x_n[15]);
//     evaluator.relinearize_inplace(x_n[15],relin_keys);
//     evaluator.rescale_to_next_inplace(x_n[15]);
//     x_n[15].scale()=scale;

//     if (DegreeK > 16){
//         evaluator.square(x_n[15],x_n[31]);
//         evaluator.relinearize_inplace(x_n[31],relin_keys);
//         evaluator.rescale_to_next_inplace(x_n[31]);
//         x_n[31].scale()=scale;
//     }


//     //compute x^3,5,9,17
//     //#pragma omp parallel for

//     for (int i = 2; i < DegreeK; i = i * 2){
//         //cout<<i+1<<" ";
//         evaluator.mod_switch_to_inplace(x_n[0],x_n[i-1].parms_id());
//         evaluator.multiply(x_n[0],x_n[i-1],x_n[i]);
//         evaluator.relinearize_inplace(x_n[i],relin_keys);
//         evaluator.rescale_to_next_inplace(x_n[i]);
//         x_n[i].scale()=scale;
//     }

//     //compute x^6,10,18
//     //#pragma omp parallel for

//     for (int i = 4; i < DegreeK; i = i * 2){
//         //cout<<i+2<<" ";
//         evaluator.mod_switch_to_inplace(x_n[1],x_n[i-1].parms_id());
//         evaluator.multiply(x_n[1],x_n[i-1],x_n[i+1]);
//         evaluator.relinearize_inplace(x_n[i+1],relin_keys);
//         evaluator.rescale_to_next_inplace(x_n[i+1]);
//         x_n[i+1].scale()=scale;
//     }

//     //compute x^7,11,19
//     //#pragma omp parallel for

//     for (int i = 4; i < DegreeK; i = i * 2){
//         //cout<<i+3<<" ";
//         evaluator.mod_switch_to_inplace(x_n[2],x_n[i-1].parms_id());
//         evaluator.multiply(x_n[2],x_n[i-1],x_n[i+2]);
//         evaluator.relinearize_inplace(x_n[i+2],relin_keys);
//         evaluator.rescale_to_next_inplace(x_n[i+2]);
//         x_n[i+2].scale()=scale;
//     }

//     //compute x^12,20, 13,21, 14,22, 15,23

//     //#pragma omp parallel for

//     for(int j = 4 ; j < 8 ; ++j){
//         for (int i = 8; i < DegreeK; i *= 2){
//             evaluator.mod_switch_to_inplace(x_n[j-1],x_n[i-1].parms_id());
//             evaluator.multiply(x_n[j-1],x_n[i-1],x_n[i+j-1]);
//             evaluator.relinearize_inplace(x_n[i+j-1],relin_keys);
//             evaluator.rescale_to_next_inplace(x_n[i+j-1]);
//             x_n[i+j-1].scale()=scale;
//         }
//     }

//     //compute x^24, 25, 26, 27, 28, 29, 30, 31
//     if (DegreeK > 16){
//         //#pragma omp parallel for

//         for (int j = 8; j < 16; ++j){
//             evaluator.mod_switch_to_inplace(x_n[j-1],x_n[15].parms_id());
//             evaluator.multiply(x_n[j-1],x_n[15],x_n[16+j-1]);
//             evaluator.relinearize_inplace(x_n[16+j-1],relin_keys);
//             evaluator.rescale_to_next_inplace(x_n[16+j-1]);
//             x_n[16+j-1].scale()=scale;
//         }
//     }




// }


// void Bootstrap_RangeCheck_PatersonStockmeyer(Ciphertext& ciphertext, const Ciphertext& input, const vector<complex<double>>& rangeCheckIndices,
//                                             const RelinKeys &relin_keys, const SEALContext& context, const SecretKey& secret_key,
//                                               const int firstDegree=32, const int secondDegree=16) {
//     //MemoryPoolHandle my_pool_larger = MemoryPoolHandle::New(true);
//     //auto old_prof_larger = MemoryManager::SwitchProfile(std::make_unique<MMProfFixed>(std::move(my_pool_larger)));

//     Decryptor decryptor(context, secret_key);
//     CKKSEncoder encoder(context);
//     Evaluator evaluator(context,encoder);

//     double scale = input.scale();
//     //cout <<log2(scale)<<endl;

//     vector<Ciphertext> kCTs(firstDegree);

//     calUptoDegreeK(kCTs, input, firstDegree, relin_keys, context);

//     for (int j = 0; j < (int) kCTs.size(); j++) {
//       evaluator.mod_switch_to_inplace(kCTs[j], kCTs[kCTs.size()-1].parms_id());
//     }

//     //cout <<"Modulus chain index for kCTs: "<< context.get_context_data(kCTs[firstDegree-1].parms_id())->chain_index()<<endl;
//     //cout <<"scale = "<<log2(kCTs[firstDegree-1].scale())<<endl;

//     //test for

//     vector<Ciphertext> kToMCTs(secondDegree);

//     calUptoDegreeK(kToMCTs, kCTs[kCTs.size()-1], secondDegree, relin_keys, context);

//     for (int j = 0; j < (int) kToMCTs.size(); j++) {
//       evaluator.mod_switch_to_inplace(kToMCTs[j], kToMCTs[kToMCTs.size()-1].parms_id());
//     }

//     //cout <<"Modulus chain index for kToMCTs: "<< context.get_context_data(kToMCTs[secondDegree-1].parms_id())->chain_index()<<endl;
//     //cout <<"scale = "<<log2(kToMCTs[secondDegree-1].scale())<<endl;
// /*
//     for (int j = 0; j < (int) kCTs.size(); j++) {
//       evaluator.mod_switch_to_inplace(kCTs[j], kToMCTs[kToMCTs.size()-1].parms_id());
//     }

//     cout <<"Modulus chain index for the result: "<< context.get_context_data(kCTs[firstDegree-1].parms_id())->chain_index()<<endl;
// */
//     Ciphertext temp_relin;
//     vector<Ciphertext> levelSum(secondDegree);

//     //#pragma omp parallel for

//     for(int i = 0; i < secondDegree; i++) {
//         //cout <<i <<endl;
//       //  Ciphertext levelSum;
//         bool flag = false;


//         for(int j = 0; j < firstDegree; j++) {
//             //cout <<j <<"- ";

//                 Plaintext plainInd;
//                 if(i*firstDegree+j+1 < 512) encoder.encode(rangeCheckIndices[i*firstDegree+j+1], kCTs[j].parms_id(), kCTs[j].scale(), plainInd);
//                 else encoder.encode(0, kCTs[j].parms_id(), kCTs[j].scale(), plainInd);

//                 if (!flag) {
//                     evaluator.multiply_plain(kCTs[j], plainInd, levelSum[i]);
//                     evaluator.rescale_to_next_inplace(levelSum[i]);

//                     flag = true;
//                 }
//                 else if(i*firstDegree+j+1 < 512) {
//                     Ciphertext tmp;
//                     evaluator.multiply_plain(kCTs[j], plainInd, tmp);
//                     evaluator.rescale_to_next_inplace(tmp);
//                     levelSum[i].scale() = scale;
//                     tmp.scale()=scale;
//                  //   cout <<log2(levelSum[i].scale())<<" "<<log2(tmp.scale())<<endl;
//                     evaluator.add_inplace(levelSum[i], tmp);
//                 }
//                 else;
//         }



//         //cout <<"Modulus chain index for levelSum[i]: "<< context.get_context_data(levelSum[i].parms_id())->chain_index()<<endl;

//         evaluator.mod_switch_to_inplace(levelSum[i], kToMCTs[i].parms_id());
//       }

//     for (int i = 0; i < secondDegree; ++i){
//         //cout <<i <<endl;
//         if(i == 0) {
//             ciphertext = levelSum[0];
//         }
//         else if (i == 1) { // initialize for temp_relin
//             evaluator.multiply(levelSum[i], kToMCTs[i - 1], temp_relin);
//             evaluator.relinearize_inplace(temp_relin, relin_keys);
//             evaluator.rescale_to_next_inplace(temp_relin);
//         }
//         else {
//             evaluator.multiply_inplace(levelSum[i], kToMCTs[i - 1]);
//             evaluator.relinearize_inplace(levelSum[i], relin_keys);
//             evaluator.rescale_to_next_inplace(levelSum[i]);
//             //cout <<log2(levelSum[i].scale())<<" "<<log2(temp_relin.scale())<<endl;
//             evaluator.add_inplace(temp_relin, levelSum[i]);
//         }

//     }

//     evaluator.mod_switch_to_inplace(ciphertext, temp_relin.parms_id());
//     //cout <<"Modulus chain index for ciphertext: "<< context.get_context_data(ciphertext.parms_id())->chain_index()<<endl;
//     //cout <<log2(ciphertext.scale())<<" "<<log2(temp_relin.scale())<<endl;
//     ciphertext.scale()=scale;
//     temp_relin.scale()=scale;
//     evaluator.add_inplace(ciphertext, temp_relin);

//     Plaintext coeff0;
//     encoder.encode(rangeCheckIndices[0],ciphertext.parms_id(),ciphertext.scale(),coeff0);
//     evaluator.add_plain_inplace(ciphertext,coeff0);





// }
