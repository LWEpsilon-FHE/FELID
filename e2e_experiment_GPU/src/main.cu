#include <cuda_runtime.h>
#include <omp.h>
#include "util/execution_timer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "phantom.h"
#include "source/bootstrapping/Bootstrapper.cuh"
#include "source/bootstrapping/common/Polynomial.cuh"
#include "source/fhe_transformer/fhe_backward_blocks.cuh"
#include "RbootSlimBootstrapper.cuh"
#include "RbootSlimPolyEval.cuh"
#include "source/fhe_transformer/fhe_pipeline.cuh"
using namespace phantom;
using namespace phantom::arith;
using namespace fhe_transformer;
namespace {
struct Environment {
    EncryptionParameters parms;
    PhantomContext context;
    PhantomSecretKey secret_key;
    PhantomPublicKey public_key;
    PhantomRelinKey relin_keys;
    PhantomGaloisKey galois_keys;
    PhantomCKKSEncoder phantom_encoder;
    moai::Encoder encoder;
    moai::Encryptor encryptor;
    moai::Decryptor decryptor;
    moai::Evaluator evaluator;

    FHEAdapterContext adapter;
    FHELevelMap level_map;
    double scale=std::ldexp(1.0,46);
    std::unique_ptr<moai::CKKSEvaluator> ckks_evaluator;
    std::unique_ptr<Bootstrapper> ordinary_bootstrapper;
    std::unique_ptr<rboot_slim::RbootSlimBootstrapper> rboot_bootstrapper;

    static PhantomContext init_context(EncryptionParameters &parms) {
        std::vector<int> bits={51};
        bits.insert(bits.end(),13,46);
        bits.insert(bits.end(),28,51);
        parms.set_poly_modulus_degree(65536);
        parms.set_coeff_modulus(CoeffModulus::Create(65536,bits));
        parms.set_secret_key_hamming_weight(192);
        parms.set_sparse_slots(32768);
        return PhantomContext(parms);
    }
    Environment():parms(scheme_type::ckks),context(init_context(parms)),secret_key(context),
        public_key(secret_key.gen_publickey(context)),relin_keys(secret_key.gen_relinkey(context)),
        galois_keys(secret_key.create_galois_keys(context)),phantom_encoder(context),
        encoder(&context,&phantom_encoder),encryptor(&context,&public_key),decryptor(&context,&secret_key),
        evaluator(&context,&phantom_encoder),level_map(FHELevelMap::from_context_top_native_depth(context,12,13)) {
        adapter.context=&context;adapter.encoder=&phantom_encoder;adapter.evaluator=&evaluator;
        adapter.ckks_encoder=&encoder;adapter.relin_keys=&relin_keys;adapter.galois_keys=&galois_keys;
        adapter.level_map=level_map;
        ckks_evaluator=std::make_unique<moai::CKKSEvaluator>(&context,&public_key,&secret_key,
            &phantom_encoder,&relin_keys,&galois_keys,scale);
        rboot_bootstrapper=std::make_unique<rboot_slim::RbootSlimBootstrapper>(
            10,15,15,30,scale,25,59,2,127,ckks_evaluator.get(),&phantom_encoder);
        rboot_bootstrapper->set_final_scale(scale);
        rboot_bootstrapper->rboot_mod_reducer->generate_sin_cos_polynomial();
        std::vector<int> rotations={0};
        for(int i=0;i<15;++i)rotations.push_back(1<<i);
        rboot_bootstrapper->addLeftRotKeys_Linear_to_vector_3(rotations);
        rboot_bootstrapper->slot_vec.push_back(15);
        rboot_bootstrapper->generate_LT_coefficient_3();
    }
    void init_ordinary_bootstrapper() {
        if(ordinary_bootstrapper)return;
        phantom::execution::Scope setup(phantom::execution::Phase::Setup);
        ordinary_bootstrapper=std::make_unique<Bootstrapper>(10,15,15,30,scale,25,79,2,1,ckks_evaluator.get());
        ordinary_bootstrapper->set_final_scale(scale);
        ordinary_bootstrapper->prepare_mod_polynomial();
        auto *reducer=ordinary_bootstrapper->mod_reducer;
        reducer->sin_cos_polynomial.constmul(NTL::to_RR(1.0/reducer->scale_inverse_coeff));
        reducer->sin_cos_polynomial.generate_poly_heap();
        std::vector<NTL::RR> inverse_coefficients(64,NTL::to_RR(0));
        NTL::RR coefficient=NTL::to_RR(1.0/(2*M_PI));
        for(int n=0;2*n+1<64;++n){inverse_coefficients[2*n+1]=coefficient;coefficient*=NTL::to_RR(double((2*n+1)*(2*n+1))/double(2*(n+1)*(2*n+3)));}
        reducer->inverse_sin_polynomial.set_polynomial(63,inverse_coefficients.data(),"power");
        reducer->inverse_sin_polynomial.generate_poly_heap_odd();reducer->inverse_deg=63;

        std::vector<int> rotations={0};
        for(int i=0;i<15;++i)rotations.push_back(1<<i);
        ordinary_bootstrapper->addLeftRotKeys_Linear_to_vector_3(rotations);
        ordinary_bootstrapper->slot_vec.push_back(15);
        ordinary_bootstrapper->generate_LT_coefficient_3();
        adamw_trim_cuda_pool_to_zero("bootstrap_init");
    }
};

PhantomCiphertext polynomial_scalar(Environment &env,const PhantomCiphertext &x,double coefficient,double scale) {
 const double prime=static_cast<double>(env.context.get_context_data(x.params_id()).parms().coeff_modulus().back().value());
 PhantomPlaintext p;env.encoder.encode(coefficient,x.params_id(),scale*prime/x.scale(),p);
 auto y=x;env.evaluator.multiply_plain_inplace(y,p);env.evaluator.rescale_to_next_inplace(y);return y;
}
void polynomial_constant(Environment &env,PhantomCiphertext &x,double coefficient) {
 PhantomPlaintext p;env.encoder.encode(coefficient,x.params_id(),x.scale(),p);env.evaluator.add_plain_inplace(x,p);
}
int polynomial_depth(Environment &env,const PhantomCiphertext &x){return env.context.get_context_data(x.params_id()).chain_depth();}
PhantomCiphertext polynomial_add(Environment &env,PhantomCiphertext a,PhantomCiphertext b,bool subtract=false) {
 int da=polynomial_depth(env,a),db=polynomial_depth(env,b);
 double target=da<=db?a.scale():b.scale();
 if(std::abs(a.scale()/b.scale()-1)>1e-12){
  if(da<=db){b=polynomial_scalar(env,b,1.,target);--db;}else{a=polynomial_scalar(env,a,1.,target);--da;}
 }
 if(da>db)env.evaluator.mod_switch_to_inplace(a,b.params_id());
 if(db>da)env.evaluator.mod_switch_to_inplace(b,a.params_id());
 if(std::abs(a.scale()/b.scale()-1)>1e-10)throw std::runtime_error("Polynomial addition scale mismatch");
 b.scale()=a.scale();if(subtract)env.evaluator.sub_inplace(a,b);else env.evaluator.add_inplace(a,b);return a;
}
PhantomCiphertext polynomial_multiply(Environment &env,PhantomCiphertext a,PhantomCiphertext b){
 int da=polynomial_depth(env,a),db=polynomial_depth(env,b);
 if(da>db)env.evaluator.mod_switch_to_inplace(a,b.params_id());if(db>da)env.evaluator.mod_switch_to_inplace(b,a.params_id());
 PhantomCiphertext y;env.evaluator.multiply(a,b,y);env.evaluator.relinearize_inplace(y,env.relin_keys);env.evaluator.rescale_to_next_inplace(y);return y;
}
PhantomCiphertext evaluate_polynomial(Environment &env,const PhantomCiphertext &z,const boot::Polynomial &p){
 const double scale=std::ldexp(1.,51);int k=p.heap_k,m=p.heap_m;
 std::vector<std::optional<PhantomCiphertext>> baby(k+1);baby[1]=z;
 for(int n=2;n<=k;++n){int a=(n+1)/2,b=n/2;auto x=polynomial_multiply(env,*baby[a],*baby[b]);env.evaluator.add_inplace(x,x);if(a==b)polynomial_constant(env,x,-1.);else x=polynomial_add(env,std::move(x),z,true);baby[n]=std::move(x);}
 std::vector<PhantomCiphertext> giant;giant.push_back(*baby[k]);
 for(int n=1;n<m;++n){auto x=polynomial_multiply(env,giant.back(),giant.back());env.evaluator.add_inplace(x,x);polynomial_constant(env,x,-1.);giant.push_back(std::move(x));}
 std::vector<std::optional<PhantomCiphertext>> heap(p.heaplen);int first=(1<<m)-1;
 int depth=0;while((1<<depth)<k)++depth;int leaf_depth=polynomial_depth(env,z)-depth-1;
 for(int i=first;i<p.heaplen;++i){if(!p.poly_heap[i])continue;auto *leaf=p.poly_heap[i];std::optional<PhantomCiphertext> value;
  for(int j=1;j<=leaf->deg;++j){double c=NTL::to_double(leaf->chebcoeff[j]);if(std::abs(c)<1e-17)continue;auto term=polynomial_scalar(env,*baby[j],c,scale);if(value)*value=polynomial_add(env,std::move(*value),std::move(term));else value=std::move(term);}
  if(!value)value=polynomial_scalar(env,z,0.,scale);
  while(polynomial_depth(env,*value)>leaf_depth)env.evaluator.mod_switch_to_next_inplace(*value);
  polynomial_constant(env,*value,NTL::to_double(leaf->chebcoeff[0]));heap[i]=std::move(value);
 }
 for(int d=m-1;d>=0;--d){for(int i=(1<<d)-1;i<(1<<(d+1))-1;++i){if(!p.poly_heap[i])continue;int l=2*i+1,r=l+1;if(!heap[l])heap[i]=std::move(heap[r]);else{auto product=polynomial_multiply(env,*heap[l],giant[m-1-d]);heap[i]=polynomial_add(env,std::move(product),*heap[r]);}}}
 return std::move(*heap[0]);
}
PhantomCiphertext inverse_iteration(Environment &env,const PhantomCiphertext &x,PhantomCiphertext y,int count) {
 auto negative_half_x=polynomial_scalar(env,x,-.5,env.scale);
 for(int i=0;i<count;++i){
  auto square=polynomial_multiply(env,y,y);
  auto cubic=polynomial_multiply(env,negative_half_x,y);
  cubic=polynomial_multiply(env,std::move(cubic),std::move(square));
  auto linear=polynomial_scalar(env,y,1.5,env.scale);
  y=polynomial_add(env,std::move(cubic),std::move(linear));
 }
 return y;
}

void ordinary_precision_bootstrap(Environment &env,PhantomCiphertext &out,PhantomCiphertext work){
 auto &b=*env.ordinary_bootstrapper;b.initial_scale=work.scale();b.modraise_inplace(work);
 work.scale()=static_cast<double>(env.context.first_context_data().parms().coeff_modulus()[0].value());
 PhantomCiphertext a,c;b.coefftoslot_full_3(a,c,work);
 auto reduce=[&](PhantomCiphertext x){
  x=evaluate_polynomial(env,x,b.mod_reducer->sin_cos_polynomial);
  for(int i=0;i<b.mod_reducer->num_double_formula;++i)b.mod_reducer->double_angle_formula(x);
  return evaluate_polynomial(env,x,b.mod_reducer->inverse_sin_polynomial);
 };
 a=reduce(std::move(a));c=reduce(std::move(c));b.slottocoeff_full_half_3(out,a,c);out.scale()=b.final_scale;
 PhantomCiphertext conjugate;env.evaluator.complex_conjugate(out,env.galois_keys,conjugate);env.evaluator.add_inplace(out,conjugate);
}

PhantomCiphertext function_sine(Environment &env,const PhantomCiphertext &input,double gain,double center,double sine_gain) {
    auto &b=*env.rboot_bootstrapper;
    auto work=input;
    if (center!=0) {
        PhantomPlaintext p;env.encoder.encode(-center,work.params_id(),work.scale(),p);
        env.evaluator.add_plain_inplace(work,p);
    }
    if (gain!=1) {
        double q=static_cast<double>(env.context.get_context_data(work.params_id()).parms().coeff_modulus().back().value());
        PhantomPlaintext p;env.encoder.encode(gain,work.params_id(),q*env.scale/work.scale(),p);
        env.evaluator.multiply_plain_inplace(work,p);env.evaluator.rescale_to_next_inplace(work);
    }
    b.initial_scale=work.scale();
    auto depth=env.context.get_context_data(work.params_id()).chain_depth();
    if (depth<3)throw std::runtime_error("Function input requires three native levels");
    while(depth>3){env.evaluator.mod_switch_to_next_inplace(work);--depth;}
    PhantomCiphertext raised,real,sine;
    b.slim_slottocoeff_full(raised,work);b.modraise_inplace(raised);
    raised.scale()=static_cast<double>(env.context.first_context_data().parms().coeff_modulus()[0].value());
    b.slim_coefftoslot_full(real,raised);
    rboot_slim::homomorphic_poly_evaluation_sparse(b.rboot_mod_reducer->sin_polynomial,*b.sparse_ops,sine,real);
    for(int i=0;i<b.rboot_mod_reducer->num_double_formula;++i)b.rboot_mod_reducer->double_angle_formula(sine);
    if(sine_gain!=1){b.sparse_ops->multiply_const_inplace(sine,sine_gain);env.evaluator.rescale_to_next_inplace(sine);}
    return sine;
}
void function_polynomial(Environment &env,PhantomCiphertext &out,const PhantomCiphertext &sine,const boot::Polynomial &poly) {
    out=evaluate_polynomial(env,sine,poly);
    out.scale()/=32.0;
    const double prime=static_cast<double>(env.context.get_context_data(out.params_id()).parms().coeff_modulus().back().value());
    PhantomPlaintext one;env.encoder.encode(1.,out.params_id(),env.scale*prime/out.scale(),one);
    env.evaluator.multiply_plain_inplace(out,one);env.evaluator.rescale_to_next_inplace(out);
    if(out.chain_index()<env.level_map.top_chain_index)::mod_switch_to_inplace(env.context,out,env.level_map.top_chain_index);
    cudaDeviceSynchronize();
}
void function_bootstrap(Environment &env,PhantomCiphertext &out,const PhantomCiphertext &input,const boot::Polynomial &poly,double gain,double center,double sine_gain) {
 auto sine=function_sine(env,input,gain,center,sine_gain);function_polynomial(env,out,sine,poly);
}

struct FunctionProfile {
 double gain,center,sine_gain;boot::Polynomial polynomial;
 FunctionProfile(const std::string &name,int degree,double g,double c,double s):gain(g),center(c),sine_gain(s) {
 phantom::execution::Scope setup(phantom::execution::Phase::Setup);
 NTL::RR::SetPrecision(2048);std::ifstream in(std::filesystem::path(FELID_DATA_DIR)/"polynomial"/(name+"_cheb.txt"));std::vector<NTL::RR> coefficients;std::string token;while(in>>token)coefficients.push_back(NTL::to_RR(token.c_str()));if(coefficients.size()!=size_t(degree+1))throw std::runtime_error("Invalid function coefficients");polynomial.set_polynomial(degree,coefficients.data(),"cheb");int k=16,m=0;while((k<<m)<=degree)++m;polynomial.generate_poly_heap_manual(k,m);
 }
};
FunctionProfile &function_profile(const std::string &name) {
 if(name=="exp"){static FunctionProfile p("exp",127,0.0078125,-64,1.5397651730468607);return p;}
 if(name=="tanh"){static FunctionProfile p("tanh",127,0.125,0,1.2936407178563822);return p;}
 if(name=="gelu"){static FunctionProfile p("gelu",255,0.015625,0,1.5397651730468607);return p;}
 if(name=="gelu_derivative"){static FunctionProfile p("gelu_derivative",255,0.015625,0,1.5397651730468607);return p;}
 throw std::runtime_error("Unknown function profile");
}

void apply_function(Environment &env,const std::string &name,PhantomCiphertext &out,const PhantomCiphertext &input) {
 phantom::execution::EncodingCacheScope encoding_cache;auto &p=function_profile(name);function_bootstrap(env,out,input,p.polynomial,p.gain,p.center,p.sine_gain);}

std::vector<std::vector<double>> load_matrix(const std::string &path) {
    std::string relative=path;
    if(relative.rfind("src/data/",0)==0)relative.erase(0,9);
    std::ifstream input(std::filesystem::path(FELID_DATA_DIR)/relative,std::ios::binary);
    std::uint64_t rows=0,cols=0;
    input.read(reinterpret_cast<char*>(&rows),sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols),sizeof(cols));
    if(!input || !rows || !cols || rows>100000 || cols>100000)
        throw std::runtime_error("Invalid matrix: "+path);
    std::vector<std::vector<double>> values(rows,std::vector<double>(cols));
    for(auto &row:values)input.read(reinterpret_cast<char*>(row.data()),cols*sizeof(double));
    if(!input || input.peek()!=std::ifstream::traits_type::eof())
        throw std::runtime_error("Invalid matrix length: "+path);
    return values;
}

struct RealLayerAssets {
    std::vector<std::vector<double>> embedded_input;
    std::vector<std::vector<std::vector<double>>> batched_input;
    std::vector<std::vector<double>> W_Q;
    std::vector<std::vector<double>> W_K;
    std::vector<std::vector<double>> W_V;
    std::vector<std::vector<double>> W_o;
    std::vector<std::vector<double>> W_fc1;
    std::vector<std::vector<double>> W_fc2;
    std::vector<double> ln1_gamma;
    std::vector<double> ln1_beta;
    std::vector<double> ln2_gamma;
    std::vector<double> ln2_beta;
    std::vector<double> bias_Q, bias_K, bias_V, bias_o, bias_fc1, bias_fc2;
};

FHEBlockTensor encrypt_real_tensor(Environment &env,
                                   const std::vector<double> &values,
                                   const std::string &name,
                                   const std::vector<int> &shape,
                                   PackingType packing)
{
    PhantomPlaintext plain;
    env.encoder.encode(values, env.context.get_first_index(), env.scale, plain);
    PhantomCiphertext ct;
    env.encryptor.encrypt(plain, ct);
    if (ct.chain_index() != env.level_map.top_chain_index)
    {
        ::mod_switch_to_inplace(env.context, ct, env.level_map.top_chain_index);
    }
    FHEBlockTensor tensor;
    tensor.cts.push_back(std::move(ct));
    tensor.name = name;
    tensor.logical_shape = shape;
    tensor.packing = packing;
    tensor.is_complex = false;
    refresh_tensor_metadata(env.context, env.level_map, tensor);
    return tensor;
}

FHEBlockTensor encrypt_complex_tensor(Environment &env,
                                      const std::vector<std::complex<double>> &values,
                                      const std::string &name,
                                      const std::vector<int> &shape,
                                      PackingType packing)
{
    PhantomPlaintext plain;
    env.encoder.encode(values, env.context.get_first_index(), env.scale, plain);
    PhantomCiphertext ct;
    env.encryptor.encrypt(plain, ct);
    if (ct.chain_index() != env.level_map.top_chain_index)
    {
        ::mod_switch_to_inplace(env.context, ct, env.level_map.top_chain_index);
    }
    FHEBlockTensor tensor;
    tensor.cts.push_back(std::move(ct));
    tensor.name = name;
    tensor.logical_shape = shape;
    tensor.packing = packing;
    tensor.is_complex = true;
    refresh_tensor_metadata(env.context, env.level_map, tensor);
    return tensor;
}

std::vector<double> decrypt_real(Environment &env, PhantomCiphertext &ct)
{
    PhantomPlaintext plain;
    env.decryptor.decrypt(ct, plain);
    std::vector<double> values;
    env.encoder.decode(plain, values);
    cudaDeviceSynchronize();
    for (double v : values) if (!std::isfinite(v))
        throw std::runtime_error("Non-finite decoded real value");
    return values;
}

FHEBlockTensor make_zero_tensor_at_level(Environment &env,
                                         const std::string &name,
                                         const std::vector<int> &shape,
                                         PackingType packing,
                                         int level,
                                         bool is_complex = false,
                                         size_t ct_count = 1)
{
    FHEBlockTensor tensor = is_complex
                                ? encrypt_complex_tensor(env,
                                                         std::vector<std::complex<double>>(env.encoder.slot_count(), {0.0, 0.0}),
                                                         name,
                                                         shape,
                                                         packing)
                                : encrypt_real_tensor(env,
                                                      std::vector<double>(env.encoder.slot_count(), 0.0),
                                                      name,
                                                      shape,
                                                      packing);
    while (tensor.cts.size() < ct_count)
    {
        tensor.cts.push_back(tensor.cts.front());
    }
    refresh_tensor_metadata(env.context, env.level_map, tensor);
    align_down_to(env.context, env.level_map, tensor, level, tensor.name);
    return tensor;
}

std::vector<std::complex<double>> decrypt_complex(Environment &env, PhantomCiphertext &ct)
{
    PhantomPlaintext plain;
    env.decryptor.decrypt(ct, plain);
    std::vector<std::complex<double>> values;
    env.encoder.decode(plain, values);
    cudaDeviceSynchronize();
    for (const auto &v : values) if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
        throw std::runtime_error("Non-finite decoded complex value");
    return values;
}

void encrypt_real_slots_at_logical_top(Environment &env,
                                       const std::vector<double> &values,
                                       PhantomCiphertext &out)
{
    for (double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite input value");
    PhantomPlaintext encoded;
    env.encoder.encode(values, env.context.get_first_index(), env.scale, encoded);
    env.encryptor.encrypt(encoded, out);
    if (out.chain_index() != env.level_map.top_chain_index)
    {
        ::mod_switch_to_inplace(env.context, out, env.level_map.top_chain_index);
    }
}

std::mutex &bootstrap_mutex()
{
    static std::mutex mutex;
    return mutex;
}

CipherUnaryCallback gain_compensated_real_bts(Environment &env,double gain)
{
    return [&env,gain](PhantomCiphertext &out,const PhantomCiphertext &input) {
        std::lock_guard<std::mutex> lock(bootstrap_mutex());
        env.init_ordinary_bootstrapper();
        phantom::execution::EncodingCacheScope encoding_cache;
        auto work=input;
        while (env.context.get_context_data(work.params_id()).chain_depth()>0)
            env.evaluator.mod_switch_to_next_inplace(work);

        const double previous=env.ordinary_bootstrapper->final_scale;
        env.ordinary_bootstrapper->set_final_scale(env.scale/gain);

        try { ordinary_precision_bootstrap(env,out,work); }
        catch (...) { env.ordinary_bootstrapper->set_final_scale(previous); throw; }
        env.ordinary_bootstrapper->set_final_scale(previous);
        out.scale()=env.scale;
        if (out.chain_index()<env.level_map.top_chain_index)
            ::mod_switch_to_inplace(env.context,out,env.level_map.top_chain_index);
        cudaDeviceSynchronize();
    };
}

PhantomBootstrappingCallbacks make_real_bootstrap_callbacks(Environment &env) {
    PhantomBootstrappingCallbacks callbacks;
    callbacks.ordinary_bts=gain_compensated_real_bts(env,1.0);
    callbacks.gain_compensated_bts=[&env](double gain){return gain_compensated_real_bts(env,gain);};
    callbacks.rboot_tanh=[&env](PhantomCiphertext &out,const PhantomCiphertext &input) {
        auto work=input;
        apply_function(env,"tanh",out,work);
        if(out.chain_index()<env.level_map.top_chain_index)
            ::mod_switch_to_inplace(env.context,out,env.level_map.top_chain_index);
        cudaDeviceSynchronize();
    };
    return callbacks;
}

CipherUnaryCallback scaled_real_bts(Environment &env, double gain)
{
    auto bootstrap = gain_compensated_real_bts(env, gain);
    return [&env, gain, bootstrap](PhantomCiphertext &out, const PhantomCiphertext &input) {
        auto work = input;
        while (env.context.get_context_data(work.params_id()).chain_depth() > 1)
            env.evaluator.mod_switch_to_next_inplace(work);
        if (env.context.get_context_data(work.params_id()).chain_depth() < 1)
            throw std::runtime_error("Gradient refresh requires one scaling level");
        const double prime = static_cast<double>(env.context.get_context_data(
            work.params_id()).parms().coeff_modulus().back().value());
        PhantomPlaintext plain;
        env.encoder.encode(gain, work.params_id(), prime * env.scale / work.scale(), plain);
        env.evaluator.multiply_plain_inplace(work, plain);
        env.evaluator.rescale_to_next_inplace(work);
        bootstrap(out, work);
    };
}

PhantomBootstrappingCallbacks make_real_exp_callbacks(Environment &env) {
    auto callbacks=make_real_bootstrap_callbacks(env);
    callbacks.exp_fb=[&env](PhantomCiphertext &out,const PhantomCiphertext &input) {
        auto work=input;
        apply_function(env,"exp",out,work);
        if(out.chain_index()<env.level_map.top_chain_index)
            ::mod_switch_to_inplace(env.context,out,env.level_map.top_chain_index);
        cudaDeviceSynchronize();
    };
    return callbacks;
}

PhantomBootstrappingCallbacks make_real_gelu_callbacks(Environment &env) {
 auto cb=make_real_bootstrap_callbacks(env);auto values=std::make_shared<FHEBlockTensor>();
 cb.gelu_derivative_component=[&env,values](FHEAdapterContext &adapter,const FHEBlockTensor &input,int level,const std::string &name,const std::string &step) {
  phantom::execution::EncodingCacheScope encoding_cache;
  if(!values->empty())throw std::runtime_error("Unconsumed GELU output");
  auto out=make_like_metadata(input,name);out.cts.resize(input.cts.size());*values=make_like_metadata(input,"gelu_output");values->cts.resize(input.cts.size());
  auto &f=function_profile("gelu"),&d=function_profile("gelu_derivative");
  for(size_t i=0;i<input.cts.size();++i){
   auto sine=function_sine(env,input.cts[i],f.gain,f.center,f.sine_gain);
   function_polynomial(env,out.cts[i],sine,d.polynomial);function_polynomial(env,values->cts[i],sine,f.polynomial);
   align_down_to(adapter.require_context(),adapter.level_map,values->cts[i],4,"gelu_output");
  }
  refresh_tensor_metadata(adapter.require_context(),adapter.level_map,*values);refresh_tensor_metadata(adapter.require_context(),adapter.level_map,out);align_down_to(adapter.require_context(),adapter.level_map,out,level,out.name);return out;
 };
 cb.gelu_output_component=[values](FHEAdapterContext &adapter,const FHEBlockTensor &input,const FHEBlockTensor &,int level,const std::string &name,const std::string &step) {
  if(values->cts.size()!=input.cts.size() || values->logical_shape!=input.logical_shape || values->level!=level)throw std::runtime_error("GELU output layout mismatch");
  auto out=std::move(*values);*values=FHEBlockTensor{};out.name=name;return out;
 };
 return cb;
}

std::vector<double> modified_col_slot_coefficients(const LayerNormParams &params,
                                                   int ct_index,
                                                   int slot_count,
                                                   const std::vector<double> &coefficients)
{
    const int n2 = slot_count / params.batch_size;
    std::vector<double> values(static_cast<size_t>(slot_count), 0.0);
    for (int j = 0; j < n2; ++j)
    {
        const int col_index = params.k * ((j / params.row) % (n2 / (params.k * params.row))) +
                              (params.k * j) / n2 +
                              ct_index * (n2 / params.row);
        if (col_index >= params.col)
        {
            continue;
        }
        const double value = coefficients[static_cast<size_t>(col_index)];
        for (int b = 0; b < params.batch_size; ++b)
        {
            values[static_cast<size_t>(j * params.batch_size + b)] = value;
        }
    }
    return values;
}

PhantomCiphertext row_reduce_sum(FHEAdapterContext &adapter,
                                 PhantomCiphertext value,
                                 const LayerNormParams &params,
                                 const std::string &step_name)
{
    const int slot_count = static_cast<int>(adapter.require_ckks_encoder().slot_count());
    int rot_index = slot_count / 2;
    const int min_index = params.row * params.batch_size;
    while (rot_index >= min_index)
    {
        PhantomCiphertext rotated;
        adapter.require_evaluator().rotate_vector(value,
                                                  rot_index,
                                                  adapter.require_galois_keys(),
                                                  rotated);
        align_scale_for_add(rotated,
                            value,
                            step_name + "/rot_" + std::to_string(rot_index),
                            step_name + "/sum");
        adapter.require_evaluator().add_inplace(value, rotated);
        
        rot_index /= 2;
    }
    return value;
}

FHEBlockTensor single_ct_tensor_like(const FHEBlockTensor &source,
                                     PhantomCiphertext ct,
                                     const std::string &name)
{
    FHEBlockTensor out = make_like_metadata(source, name);
    out.cts.clear();
    out.cts.push_back(std::move(ct));
    out.is_complex = false;
    return out;
}

PhantomPlaintext encode_real_vector_like(FHEAdapterContext &adapter,
                                         const PhantomCiphertext &ct,
                                         const std::vector<double> &values)
{
    PhantomPlaintext plain;
    encode_values(adapter, values, ct.params_id(), ct.scale(), plain);
    return plain;
}

FHEBlockTensor refresh_tensor_up_to_level(FHEAdapterContext &adapter,
                                          const FHEBlockTensor &input,
                                          const PhantomBootstrappingCallbacks &callbacks,
                                          int target_level,
                                          const std::string &out_name,
                                          const std::string &step_name)
{
    FHEBlockTensor work = input;
    work.name = out_name + "_work";
    if (work.level > target_level)
    {
        align_down_to(adapter.require_context(), adapter.level_map, work, target_level, work.name);
        work.name = out_name;
        return work;
    }
    if (work.level == target_level)
    {
        work.name = out_name;
        return work;
    }
    if (work.level <= 0 || target_level >= adapter.level_map.max_logical_level)
        throw std::runtime_error(step_name + " scaled cache refresh needs one input/output level");

    constexpr double cache_scale = 1.0 / 8.0;
    for (size_t i = 0; i < work.cts.size(); ++i) {
        auto p = encode_real_constant_like(adapter, work.cts[i], cache_scale);
        work.cts[i] = multiply_plain_and_rescale(adapter, work.cts[i], p,
                                               step_name + "_scale_cache");
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, work);
    if (work.level > 0)
    {
        align_down_to(adapter.require_context(), adapter.level_map, work, 0, work.name);
    }
    FHEBlockTensor refreshed = ordinary_bts(adapter,
                                            work,
                                            callbacks,
                                            out_name,
                                            target_level + 1,
                                            step_name + "_ordinary_BTS_refresh");
    for (size_t i = 0; i < refreshed.cts.size(); ++i) {
        auto p = encode_real_constant_like(adapter, refreshed.cts[i], 1.0 / cache_scale);
        refreshed.cts[i] = multiply_plain_and_rescale(adapter, refreshed.cts[i], p,
                                                    step_name + "_restore_cache_scale");
    }
    refresh_tensor_metadata(adapter.require_context(), adapter.level_map, refreshed);
    refreshed.logical_shape = input.logical_shape;
    refreshed.packing = input.packing;
    refreshed.is_complex = input.is_complex;
    return refreshed;
}

LayerNormBackend make_real_layernorm_backend(Environment &env)
{

    return [&env](FHEAdapterContext &adapter,
                  const FHEBlockTensor &input,
                  const LayerNormParams &params,
                  const std::string &step_name) {
        const PhantomBootstrappingCallbacks callbacks = make_real_bootstrap_callbacks(env);
        if (!callbacks.ordinary_bts)
        {
            throw std::runtime_error(step_name + " requires real ordinary BTS callback");
        }

        const std::vector<double> gamma =
            params.gamma.empty() ? std::vector<double>(params.col, 1.0) : params.gamma;
        const std::vector<double> beta =
            params.beta.empty() ? std::vector<double>(params.col, 0.0) : params.beta;
        std::vector<double> gamma_scaled = gamma;
        const double inv_sqrt_c = 1.0 / std::sqrt(params.variance_c);
        for (double &value : gamma_scaled)
        {
            value *= inv_sqrt_c;
        }

        const int expected_no_norm_level =
            params.expected_no_norm_level >= 0 ? params.expected_no_norm_level : params.expected_output_level;
        const int expected_inv_level =
            params.expected_inv_sqrt_var_level >= 0 ? params.expected_inv_sqrt_var_level : params.expected_output_level;
        const int slot_count = static_cast<int>(adapter.require_ckks_encoder().slot_count());

        PhantomCiphertext row_sum = input.cts.front();
        for (size_t i = 1; i < input.cts.size(); ++i)
        {
            PhantomCiphertext addend = input.cts[i];
            align_pair_for_add(adapter.require_context(), adapter.level_map, row_sum, addend, step_name + "/row_sum", input.name + "/ct_" + std::to_string(i));
            align_scale_for_add(addend,
                                row_sum,
                                input.name + "/ct_" + std::to_string(i),
                                step_name + "/row_sum");
            adapter.require_evaluator().add_inplace(row_sum, addend);

        }
        row_sum = row_reduce_sum(adapter, row_sum, params, step_name + "/mean_row_sum");

        std::vector<double> inv_col_values(static_cast<size_t>(slot_count),
                                           1.0 / static_cast<double>(params.col));
        PhantomPlaintext inv_col_plain =
            encode_real_vector_like(adapter, row_sum, inv_col_values);
        PhantomCiphertext mean =
            multiply_plain_and_rescale(adapter,
                                       row_sum,
                                       inv_col_plain,
                                       step_name + "/mean_inv_col");

        FHEBlockTensor no_norm_raw = make_like_metadata(input, step_name + "_ln_no_norm_raw");
        no_norm_raw.is_complex = false;
        no_norm_raw.cts.resize(input.cts.size());
        for (size_t i = 0; i < input.cts.size(); ++i)
        {
            PhantomCiphertext centered = input.cts[i];
            PhantomCiphertext mean_copy = mean;
            align_pair_for_add(adapter.require_context(), adapter.level_map, centered, mean_copy, input.name + "/ct_" + std::to_string(i), step_name + "/mean");
            align_scale_for_add(mean_copy,
                                centered,
                                step_name + "/mean",
                                input.name + "/ct_" + std::to_string(i));
            adapter.require_evaluator().sub(centered, mean_copy, no_norm_raw.cts[i]);

        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, no_norm_raw);

        FHEBlockTensor no_norm_cache =
            refresh_tensor_up_to_level(adapter,
                                       no_norm_raw,
                                       callbacks,
                                       expected_no_norm_level,
                                       "ln_no_norm_real",
                                       step_name + "_ln_no_norm_cache");

        PhantomCiphertext var_sum;
        bool var_set = false;
        for (size_t i = 0; i < no_norm_raw.cts.size(); ++i)
        {
            PhantomCiphertext centered = no_norm_raw.cts[i];
            PhantomCiphertext squared;
            adapter.require_evaluator().square(centered, squared);

            if (!var_set)
            {
                var_sum = std::move(squared);
                var_set = true;
            }
            else
            {
                align_scale_for_add(squared,
                                    var_sum,
                                    step_name + "/variance_square_" + std::to_string(i),
                                    step_name + "/variance_sum");
                adapter.require_evaluator().add_inplace(var_sum, squared);

            }
        }
        adapter.require_evaluator().relinearize_inplace(var_sum, adapter.require_relin_keys());
        adapter.require_evaluator().rescale_to_next_inplace(var_sum);

        const bool self_output_layernorm = params.expected_input_level >= 7;
        if (self_output_layernorm && get_level(adapter.level_map, var_sum) > 1)
        {
            align_down_to(adapter.require_context(), adapter.level_map, var_sum, 1, step_name + "/variance_sum_pre_reduce");
        }
        var_sum = row_reduce_sum(adapter, var_sum, params, step_name + "/variance_row_sum");

        PhantomCiphertext var_for_inv;
        if (self_output_layernorm)
        {
            std::vector<double> inv_cn_values(static_cast<size_t>(slot_count),
                                              1.0 / (static_cast<double>(params.col) * params.variance_c));
            PhantomPlaintext inv_cn_plain =
                encode_real_vector_like(adapter, var_sum, inv_cn_values);
            PhantomCiphertext var_scaled =
                multiply_plain_and_rescale(adapter,
                                           var_sum,
                                           inv_cn_plain,
                                           step_name + "/variance_inv_cn");
            FHEBlockTensor var_low =
                single_ct_tensor_like(input,
                                      std::move(var_scaled),
                                      step_name + "_variance_low");
            refresh_tensor_metadata(adapter.require_context(), adapter.level_map, var_low);
            align_down_to(adapter.require_context(), adapter.level_map, var_low, 0, var_low.name);
            FHEBlockTensor var_refreshed =
                ordinary_bts(adapter,
                             var_low,
                             callbacks,
                             step_name + "_variance_for_invsqrt",
                             adapter.level_map.max_logical_level,
                             step_name + "_variance_ordinary_BTS");
            var_for_inv = var_refreshed.cts.front();
        }
        else
        {
            std::vector<double> inv_cn_values(static_cast<size_t>(slot_count),
                                              1.0 / (static_cast<double>(params.col) * params.variance_c));
            if (get_level(adapter.level_map, var_sum) > 0)
            {
                PhantomPlaintext inv_cn_plain =
                    encode_real_vector_like(adapter, var_sum, inv_cn_values);
                PhantomCiphertext var_scaled =
                    multiply_plain_and_rescale(adapter,
                                               var_sum,
                                               inv_cn_plain,
                                               step_name + "/variance_inv_cn_pre_bts");
                FHEBlockTensor var_low =
                    single_ct_tensor_like(input,
                                          std::move(var_scaled),
                                          step_name + "_variance_low");
                refresh_tensor_metadata(adapter.require_context(), adapter.level_map, var_low);
                align_down_to(adapter.require_context(), adapter.level_map, var_low, 0, var_low.name);
                FHEBlockTensor var_refreshed =
                    ordinary_bts(adapter,
                                 var_low,
                                 callbacks,
                                 step_name + "_variance_for_invsqrt",
                                 adapter.level_map.max_logical_level,
                                 step_name + "_variance_ordinary_BTS");
                var_for_inv = var_refreshed.cts.front();
                align_down_to(adapter.require_context(), adapter.level_map, var_for_inv, adapter.level_map.max_logical_level - 2, step_name + "/variance_pre_invsqrt_boot2");
            }
            else
            {
                
                FHEBlockTensor var_low =
                    single_ct_tensor_like(input,
                                          std::move(var_sum),
                                          step_name + "_variance_low");
                refresh_tensor_metadata(adapter.require_context(), adapter.level_map, var_low);
                align_down_to(adapter.require_context(), adapter.level_map, var_low, 0, var_low.name);
                auto variance_callbacks=callbacks;
                variance_callbacks.ordinary_bts=scaled_real_bts(env,1.0/(params.col*params.variance_c));
                FHEBlockTensor var_refreshed =
                    ordinary_bts(adapter,
                                 var_low,
                                 variance_callbacks,
                                 step_name + "_variance_for_scale",
                                 adapter.level_map.max_logical_level,
                                 step_name + "_variance_ordinary_BTS");
                var_for_inv = var_refreshed.cts.front();
                align_down_to(adapter.require_context(), adapter.level_map, var_for_inv, adapter.level_map.max_logical_level - 1, step_name + "/variance_pre_inv_cn");
                PhantomPlaintext inv_cn_plain =
                    encode_real_vector_like(adapter, var_for_inv, inv_cn_values);
                var_for_inv =
                    multiply_plain_and_rescale(adapter,
                                               var_for_inv,
                                               inv_cn_plain,
                                               step_name + "/variance_inv_cn");
            }
        }

        auto variance_epsilon = encode_real_constant_like(adapter, var_for_inv,
                                                          params.epsilon / params.variance_c);
        adapter.require_evaluator().add_plain_inplace(var_for_inv, variance_epsilon);

        const int inv_bootstrap_return_level =
            self_output_layernorm ? adapter.level_map.max_logical_level : 8;
        auto inv_bootstrap = [&adapter,
                              callbacks,
                              inv_bootstrap_return_level,
                              &input,
                              &step_name](PhantomCiphertext &out,
                                           const PhantomCiphertext &in) {
            (void)input;
            callbacks.ordinary_bts(out, in);
            const int callback_level = get_level(adapter.level_map, out);
            if (callback_level != adapter.level_map.max_logical_level)
            {
                throw std::runtime_error(step_name + " inv_sqrt internal BTS returned level " +
                                         std::to_string(callback_level) + ", expected " +
                                         std::to_string(adapter.level_map.max_logical_level));
            }

            if (inv_bootstrap_return_level < adapter.level_map.max_logical_level)
            {
                align_down_to(adapter.require_context(), adapter.level_map, out, inv_bootstrap_return_level, step_name + "/invsqrt_boot_return");
            }
            
        };

        auto initial=polynomial_scalar(env,var_for_inv,-2.,env.scale);
        polynomial_constant(env,initial,3.);
        auto inverse_estimate=inverse_iteration(env,var_for_inv,std::move(initial),4);
        PhantomCiphertext inverse_refreshed;
        scaled_real_bts(env,1./8)(inverse_refreshed,inverse_estimate);
        PhantomCiphertext inv_sqrt=inverse_iteration(env,var_for_inv,std::move(inverse_refreshed),2);
        cudaDeviceSynchronize();

        FHEBlockTensor inv_sqrt_for_output =
            single_ct_tensor_like(input,
                                  std::move(inv_sqrt),
                                  "ln_inv_sqrt_var_real_compute");
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, inv_sqrt_for_output);

        FHEBlockTensor inv_sqrt_tensor = inv_sqrt_for_output;
        inv_sqrt_tensor.name = "ln_inv_sqrt_var_real";
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, inv_sqrt_tensor);
        if (inv_sqrt_tensor.level < expected_inv_level)
        {
            inv_sqrt_tensor =
                refresh_tensor_up_to_level(adapter,
                                           inv_sqrt_tensor,
                                           callbacks,
                                           expected_inv_level,
                                           "ln_inv_sqrt_var_real",
                                           step_name + "_inv_sqrt_cache");
        }
        else if (inv_sqrt_tensor.level > expected_inv_level)
        {
            align_down_to(adapter.require_context(), adapter.level_map, inv_sqrt_tensor, expected_inv_level, inv_sqrt_tensor.name);
        }

        FHEBlockTensor output = make_like_metadata(input, "ln_output_real");
        output.is_complex = false;
        output.cts.resize(input.cts.size());
        for (size_t i = 0; i < no_norm_raw.cts.size(); ++i)
        {
            const std::vector<double> gamma_slots =
                modified_col_slot_coefficients(params,
                                               static_cast<int>(i),
                                               slot_count,
                                               gamma_scaled);
            PhantomCiphertext affine_centered;
            if (self_output_layernorm)
            {
                PhantomCiphertext x_scaled = input.cts[i];
                PhantomPlaintext gamma_plain =
                    encode_real_vector_like(adapter, x_scaled, gamma_slots);
                x_scaled =
                    multiply_plain_and_rescale(adapter,
                                               x_scaled,
                                               gamma_plain,
                                               step_name + "/gamma_x_" + std::to_string(i));

                std::vector<double> gamma_over_col = gamma_slots;
                for (double &value : gamma_over_col)
                {
                    value /= static_cast<double>(params.col);
                }
                PhantomCiphertext sum_scaled = row_sum;
                PhantomPlaintext gamma_sum_plain =
                    encode_real_vector_like(adapter, sum_scaled, gamma_over_col);
                sum_scaled =
                    multiply_plain_and_rescale(adapter,
                                               sum_scaled,
                                               gamma_sum_plain,
                                               step_name + "/gamma_sum_" + std::to_string(i));

                align_pair_for_add(adapter.require_context(), adapter.level_map, x_scaled, sum_scaled, step_name + "/gamma_x_" + std::to_string(i), step_name + "/gamma_sum_" + std::to_string(i));
                align_scale_for_add(sum_scaled,
                                    x_scaled,
                                    step_name + "/gamma_sum_" + std::to_string(i),
                                    step_name + "/gamma_x_" + std::to_string(i));
                adapter.require_evaluator().sub(x_scaled, sum_scaled, affine_centered);

            }
            else
            {
                PhantomCiphertext inv_gamma = inv_sqrt_for_output.cts.front();
                PhantomPlaintext gamma_plain =
                    encode_real_vector_like(adapter, inv_gamma, gamma_slots);
                inv_gamma =
                    multiply_plain_and_rescale(adapter,
                                               inv_gamma,
                                               gamma_plain,
                                               step_name + "/gamma_" + std::to_string(i));

                PhantomCiphertext centered = no_norm_raw.cts[i];
                align_pair_for_mul(adapter.require_context(), adapter.level_map, centered, inv_gamma, no_norm_raw.name + "/ct_" + std::to_string(i), step_name + "/inv_gamma_" + std::to_string(i));
                align_scale_for_add(centered,
                                    inv_gamma,
                                    no_norm_raw.name + "/ct_" + std::to_string(i),
                                    step_name + "/inv_gamma_" + std::to_string(i));
                adapter.require_evaluator().multiply(centered, inv_gamma, affine_centered);
                adapter.require_evaluator().relinearize_inplace(affine_centered, adapter.require_relin_keys());
                adapter.require_evaluator().rescale_to_next_inplace(affine_centered);

                const std::vector<double> beta_slots =
                    modified_col_slot_coefficients(params,
                                                   static_cast<int>(i),
                                                   slot_count,
                                                   beta);
                PhantomPlaintext beta_plain =
                    encode_real_vector_like(adapter, affine_centered, beta_slots);
                adapter.require_evaluator().add_plain_inplace(affine_centered, beta_plain);

                output.cts[i] = std::move(affine_centered);
                continue;
            }

            PhantomCiphertext inv = inv_sqrt_for_output.cts.front();
            align_pair_for_mul(adapter.require_context(), adapter.level_map, affine_centered, inv, step_name + "/affine_centered_" + std::to_string(i), inv_sqrt_for_output.name + "/ct_0");
            align_scale_for_add(affine_centered,
                                inv,
                                step_name + "/affine_centered_" + std::to_string(i),
                                inv_sqrt_for_output.name + "/ct_0");
            adapter.require_evaluator().multiply(affine_centered, inv, output.cts[i]);
            adapter.require_evaluator().relinearize_inplace(output.cts[i], adapter.require_relin_keys());
            adapter.require_evaluator().rescale_to_next_inplace(output.cts[i]);

            const std::vector<double> beta_slots =
                modified_col_slot_coefficients(params,
                                               static_cast<int>(i),
                                               slot_count,
                                               beta);
            PhantomPlaintext beta_plain =
                encode_real_vector_like(adapter, output.cts[i], beta_slots);
            adapter.require_evaluator().add_plain_inplace(output.cts[i], beta_plain);

        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, output);

        LayerNormResult result;
        result.no_norm = std::move(no_norm_cache);
        result.no_norm.name = "ln_no_norm_real";
        result.inv_sqrt_var = std::move(inv_sqrt_tensor);
        result.inv_sqrt_var.name = "ln_inv_sqrt_var_real";
        FHEBlockTensor inv_std = result.inv_sqrt_var;
        for (auto &ct : inv_std.cts) {
            auto p = encode_real_constant_like(adapter, ct, inv_sqrt_c);
            ct = multiply_plain_and_rescale(adapter, ct, p, step_name + "_unscale_inverse_stddev");
        }
        refresh_tensor_metadata(adapter.require_context(), adapter.level_map, inv_std);
        result.inverse_stddev = refresh_tensor_up_to_level(
            adapter, inv_std, callbacks, expected_inv_level,
            "ln_inverse_stddev_real", step_name + "_inverse_stddev_cache");
        FHEBlockTensor inv_repeated = result.inverse_stddev;
        while (inv_repeated.cts.size() < result.no_norm.cts.size())
            inv_repeated.cts.push_back(inv_repeated.cts.front());
        auto normalized = fhe_hadamard(adapter, result.no_norm, inv_repeated,
                                       "ln_normalized_pre_refresh", step_name + "_normalize_cache");
        result.normalized_input = refresh_tensor_up_to_level(
            adapter, normalized, callbacks, expected_no_norm_level,
            "ln_normalized_real", step_name + "_normalized_cache");
        result.output = std::move(output);
        result.output.name = "ln_output_real";
        return result;
    };
}

AdamWHEInvSqrtBackend make_real_adamw_inv_sqrt_backend(Environment &env) {
 phantom::execution::Scope setup(phantom::execution::Phase::Setup);
 NTL::RR::SetPrecision(2048);
 std::ifstream in(std::filesystem::path(FELID_DATA_DIR)/"polynomial/inverse_sqrt_cheb.txt");
 std::vector<NTL::RR> coefficients;std::string token;
 while(in>>token)coefficients.push_back(NTL::to_RR(token.c_str()));
 if(coefficients.size()!=128)throw std::runtime_error("Invalid inverse square root polynomial");
 auto initial=std::make_shared<boot::Polynomial>(127,coefficients.data(),"cheb");
 initial->generate_poly_heap_manual(16,3);
 return [&env,initial](FHEAdapterContext &adapter,FHEBlockTensor input,const AdamWHEConfig &config,const std::string &step) {
  auto output=make_like_metadata(input,"adamw_inverse");output.cts.resize(input.cts.size());
  const int polynomial_level=adapter.level_map.max_logical_level;
  const auto polynomial_chain=adapter.level_map.chain_index_for_logical_level(env.context,polynomial_level);
  const auto input_chain=adapter.level_map.chain_index_for_logical_level(env.context,config.input_level);
  auto refresh=[&env,input_chain](const PhantomCiphertext &value){PhantomCiphertext out;scaled_real_bts(env,1./128)(out,value);if(out.chain_index()<input_chain)::mod_switch_to_inplace(env.context,out,input_chain);return out;};
  for(size_t i=0;i<input.cts.size();++i){
   auto x=input.cts[i];
   if(get_level(adapter.level_map,x)<polynomial_level){PhantomCiphertext refreshed;make_real_bootstrap_callbacks(env).ordinary_bts(refreshed,x);x=std::move(refreshed);}
   if(x.chain_index()<polynomial_chain)::mod_switch_to_inplace(env.context,x,polynomial_chain);
   auto z=polynomial_scalar(env,x,2.,env.scale);polynomial_constant(env,z,-1.);
   PhantomCiphertext y;initial->homomorphic_poly_evaluation(env.ckks_evaluator.get(),y,z);
   if(x.chain_index()<input_chain)::mod_switch_to_inplace(env.context,x,input_chain);
   for(int iteration=0;iteration<3;++iteration){
    if(get_level(adapter.level_map,y)<3)y=refresh(y);
    y=env.ckks_evaluator->invert_sqrt_unit_newton_with_initial(x,y,1);
   }
   if(get_level(adapter.level_map,y)<config.inv_sqrt_output_level)y=refresh(y);
   output.cts[i]=std::move(y);
  }
  refresh_tensor_metadata(env.context,env.level_map,output);align_down_to(env.context,env.level_map,output,config.inv_sqrt_output_level,output.name);return output;
 };
}

void require_matrix_shape(const std::string &name,
                          const std::vector<std::vector<double>> &matrix,
                          int rows,
                          int cols)
{
    if (static_cast<int>(matrix.size()) != rows ||
        matrix.empty() ||
        static_cast<int>(matrix.front().size()) != cols)
    {
        std::ostringstream oss;
        oss << name << " shape mismatch actual=["
            << matrix.size() << ","
            << (matrix.empty() ? 0 : matrix.front().size())
            << "] expected=[" << rows << "," << cols << "]";
        throw std::runtime_error(oss.str());
    }
}

std::vector<double> matrix_to_vector(const std::vector<std::vector<double>> &matrix,
                                         const std::string &name,
                                         int expected_size)
{
    std::vector<double> values;
    if (matrix.size() == 1)
    {
        values = matrix.front();
    }
    else if (!matrix.empty() && matrix.front().size() == 1)
    {
        values.reserve(matrix.size());
        for (const auto &row : matrix)
        {
            values.push_back(row.front());
        }
    }
    else
    {
        throw std::runtime_error(name + " is not a vector matrix");
    }
    if (static_cast<int>(values.size()) != expected_size)
    {
        throw std::runtime_error(name + " vector length mismatch");
    }
    return values;
}

std::vector<std::vector<std::vector<double>>> replicate_batch(
    const std::vector<std::vector<double>> &matrix,
    int batch_size)
{
    return std::vector<std::vector<std::vector<double>>>(
        batch_size,
        matrix);
}

FHEBlockTensor class1_packing_batch_encrypt(Environment &env,
                                            const std::vector<std::vector<double>> &W,
                                            int batch_size,
                                            int row_X,
                                            int num_row,
                                            int num_col,
                                            const std::string &name)
{
    const int n = static_cast<int>(env.encoder.slot_count());
    const int num_ct = num_row * num_col * batch_size / n;
    const int n2 = n / (batch_size * row_X);
    if (num_ct <= 0 ||
        num_row * num_col * batch_size % n != 0 ||
        num_col % row_X != 0 ||
        num_row % n2 != 0)
    {
        throw std::runtime_error("class1_packing_batch_encrypt dimensions do not match CPU Class_1_packing_batch");
    }
    require_matrix_shape(name + "_W", W, num_row, num_col);

    FHEBlockTensor output;
    output.name = name;
    output.logical_shape = {num_row, num_col};
    output.packing = PackingType::RepeatedColumn;
    output.is_complex = false;
    output.cts.resize(static_cast<size_t>(num_ct));

    const int col_tiles = num_col / row_X;
    const int row_tiles = num_row / n2;
    for (int i = 0; i < col_tiles; ++i)
    {
        for (int j = 0; j < row_tiles; ++j)
        {
            std::vector<double> vec(static_cast<size_t>(n), 0.0);
            for (int ind1 = 0; ind1 < n2; ++ind1)
            {
                for (int ind2 = 0; ind2 < row_X; ++ind2)
                {
                    for (int ind3 = 0; ind3 < batch_size; ++ind3)
                    {
                        vec[static_cast<size_t>(ind1 * row_X * batch_size + ind2 * batch_size + ind3)] =
                            W[static_cast<size_t>(j * n2 + ind1)][static_cast<size_t>(i * row_X + ind2)];
                    }
                }
            }
            encrypt_real_slots_at_logical_top(env, vec, output.cts[static_cast<size_t>(i * row_tiles + j)]);
        }
    }
    refresh_tensor_metadata(env.context, env.level_map, output);
    return output;
}

FHEBlockTensor class2_packing_batch_encrypt(Environment &env,
                                            const std::vector<std::vector<double>> &W,
                                            int batch_size,
                                            int num_row,
                                            int num_col,
                                            const std::string &name)
{
    if (num_col != 2)
    {
        throw std::runtime_error("class2_packing_batch_encrypt currently mirrors CPU Class_2_packing_batch for 2 columns");
    }
    require_matrix_shape(name + "_W", W, num_row, num_col);
    const int n = static_cast<int>(env.encoder.slot_count());
    FHEBlockTensor output;
    output.name = name;
    output.logical_shape = {num_row, num_col};
    output.packing = PackingType::InterleavedRow;
    output.is_complex = false;
    output.cts.resize(1);

    std::vector<double> vec(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < num_row; ++j)
        {
            for (int b = 0; b < batch_size; ++b)
            {
                vec[static_cast<size_t>(i * n / 2 + j * batch_size + b)] =
                    W[static_cast<size_t>(j)][static_cast<size_t>(i)];
            }
        }
    }

    encrypt_real_slots_at_logical_top(env, vec, output.cts.front());
    refresh_tensor_metadata(env.context, env.level_map, output);
    return output;
}

RealLayerAssets load_real_layer_assets(int layer_index,
                                       int batch_size,
                                       int row,
                                       int hidden,
                                       int ffn)
{
    const std::string base = "src/data/layer_" + std::to_string(layer_index);
    RealLayerAssets assets;
    if(layer_index==0){assets.embedded_input = load_matrix(base + "/Attention/BertSelfAttention/allresults/embedded_inputs.bin");
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_embedded_inputs",
                         assets.embedded_input,
                         row,
                         hidden);
    assets.batched_input = replicate_batch(assets.embedded_input, batch_size);

    }
assets.W_Q = transpose_matrix(load_matrix(base + "/Attention/BertSelfAttention/parms/query_weight.bin"));
    assets.W_K = transpose_matrix(load_matrix(base + "/Attention/BertSelfAttention/parms/key_weight.bin"));
    assets.W_V = transpose_matrix(load_matrix(base + "/Attention/BertSelfAttention/parms/value_weight.bin"));
    assets.W_o = transpose_matrix(load_matrix(base + "/Attention/SelfOutput/parms/self_output_dense_weight.bin"));
    assets.W_fc1 = transpose_matrix(load_matrix(base + "/Intermediate/parms/intermediate_dense_weight.bin"));
    assets.W_fc2 = transpose_matrix(load_matrix(base + "/Output/parms/final_output_dense_weight.bin"));
    auto bias = [&](const std::string &path, int size) {
        return matrix_to_vector(load_matrix(base + path), path, size);
    };
    assets.bias_Q = bias("/Attention/BertSelfAttention/parms/query_bias.bin", hidden);
    assets.bias_K = bias("/Attention/BertSelfAttention/parms/key_bias.bin", hidden);
    assets.bias_V = bias("/Attention/BertSelfAttention/parms/value_bias.bin", hidden);
    assets.bias_o = bias("/Attention/SelfOutput/parms/self_output_dense_bias.bin", hidden);
    assets.bias_fc1 = bias("/Intermediate/parms/intermediate_dense_bias.bin", ffn);
    assets.bias_fc2 = bias("/Output/parms/final_output_dense_bias.bin", hidden);

    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_Q", assets.W_Q, hidden, hidden);
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_K", assets.W_K, hidden, hidden);
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_V", assets.W_V, hidden, hidden);
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_o", assets.W_o, hidden, hidden);
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_fc1", assets.W_fc1, hidden, ffn);
    require_matrix_shape("layer_" + std::to_string(layer_index) + "_W_fc2", assets.W_fc2, ffn, hidden);

    assets.ln1_gamma =
        matrix_to_vector(load_matrix(base + "/Attention/SelfOutput/parms/self_output_LayerNorm_weight.bin"),
                             "layer_" + std::to_string(layer_index) + "_ln1_gamma",
                             hidden);
    assets.ln1_beta =
        matrix_to_vector(load_matrix(base + "/Attention/SelfOutput/parms/self_output_LayerNorm_bias.bin"),
                             "layer_" + std::to_string(layer_index) + "_ln1_beta",
                             hidden);
    assets.ln2_gamma =
        matrix_to_vector(load_matrix(base + "/Output/parms/final_output_LayerNorm_weight.bin"),
                             "layer_" + std::to_string(layer_index) + "_ln2_gamma",
                             hidden);
    assets.ln2_beta =
        matrix_to_vector(load_matrix(base + "/Output/parms/final_output_LayerNorm_bias.bin"),
                             "layer_" + std::to_string(layer_index) + "_ln2_beta",
                             hidden);
    return assets;
}

LayerInputs make_real_shape_layer_inputs(Environment &env,
                                                         const RealLayerAssets &assets,
                                                         const std::string &prefix,
                                                         int batch,
                                                         int row,
                                                         int hidden,
                                                         int rank,
                                                         int k_step,
                                                         bool encrypt_x = true)
{
    LayerInputs inputs;
    if (encrypt_x)
    {
        inputs.X = modified_col_packing_batch_encrypt(assets.batched_input,
                                                      batch,
                                                      row,
                                                      hidden,
                                                      k_step,
                                                      env.scale,
                                                      env.context,
                                                      env.level_map,
                                                      env.encoder,
                                                      env.encryptor,
                                                      prefix + "_X");
    }
    inputs.W_Q = assets.W_Q;
    inputs.W_K = assets.W_K;
    inputs.W_V = assets.W_V;
    inputs.W_o = assets.W_o;
    inputs.W_fc1 = assets.W_fc1;
    inputs.W_fc2 = assets.W_fc2;

    const int x_ct_count = row * hidden * batch / static_cast<int>(env.encoder.slot_count());
    const int lora_b_ct_count = rank * x_ct_count / 2;
    inputs.B_Q = make_zero_tensor_at_level(env, prefix + "_B_Q", {rank, hidden},
                                           PackingType::InterleavedColumnComplex, 12, true, lora_b_ct_count);
    inputs.B_K = make_zero_tensor_at_level(env, prefix + "_B_K", {rank, hidden},
                                           PackingType::InterleavedColumnComplex, 12, true, lora_b_ct_count);
    inputs.B_V = make_zero_tensor_at_level(env, prefix + "_B_V", {rank, hidden},
                                           PackingType::InterleavedColumnComplex, 12, true, lora_b_ct_count);

        std::vector<std::vector<double>> A(hidden, std::vector<double>(rank));
        for (int i = 0; i < hidden; ++i)
            for (int j = 0; j < rank; ++j) A[i][j] = 0.01 * std::sin(0.37*i + 1.13*j + 0.5);
        inputs.A_Q = lora_a_packing_batch_encrypt(A, batch, row, hidden, rank, env.scale,
            env.context, env.level_map, env.encoder, env.encryptor, prefix + "_A_Q");
        inputs.A_K = inputs.A_Q; inputs.A_K.name = prefix + "_A_K";
        inputs.A_V = inputs.A_Q; inputs.A_V.name = prefix + "_A_V";

    inputs.backward_inputs.W_o_plaintext = inputs.W_o;
    inputs.backward_inputs.W_Q_plaintext = inputs.W_Q;
    inputs.backward_inputs.W_K_plaintext = inputs.W_K;
    inputs.backward_inputs.W_V_plaintext = inputs.W_V;
    inputs.backward_inputs.W_fc1_plaintext = inputs.W_fc1;
    inputs.backward_inputs.W_fc2_plaintext = inputs.W_fc2;
    return inputs;
}

PipelineConfigs make_real_shape_configs(Environment &env,
                                             const RealLayerAssets &assets,
                                             int batch,
                                             int row,
                                             int hidden,
                                             int ffn,
                                             int rank,
                                             int num_heads,
                                             int head_dim,
                                             int k_step)
{
    PipelineConfigs configs;
    configs.f1.bias_Q = assets.bias_Q;
    configs.f1.bias_K = assets.bias_K;
    configs.f1.bias_V = assets.bias_V;
    configs.f3.bias = assets.bias_o;
    configs.f4.bias_fc1 = assets.bias_fc1;
    configs.f4.bias_fc2 = assets.bias_fc2;

    PhantomBootstrappingCallbacks real_bootstrap_callbacks;

        real_bootstrap_callbacks = make_real_bootstrap_callbacks(env);

    configs.f1.callbacks = real_bootstrap_callbacks;
    configs.f1.qkv_pcmm_params = {batch, hidden, row, hidden, hidden, k_step};
    configs.f1.lora_a_params = {batch, hidden, row, rank, hidden, k_step};
    configs.f1.lora_b_params = {batch, rank, row, hidden, rank, k_step};

    configs.f2.callbacks = make_real_exp_callbacks(env);
    configs.f2.cache_backward_auxiliary = true;
    configs.f2.num_heads = num_heads;
    configs.f2.head_dim = head_dim;
    configs.f2.kernel_params = {batch, row, head_dim, k_step};
    configs.f2.value_params = {batch, row, head_dim, row, k_step};

    configs.f3.wo_params = {batch, hidden, row, hidden, hidden, k_step};

    configs.ln1.params.row = row;
    configs.ln1.params.col = hidden;
    configs.ln1.params.batch_size = batch;
    configs.ln1.params.k = k_step;
    configs.ln1.params.gamma = assets.ln1_gamma;
    configs.ln1.params.beta = assets.ln1_beta;
    configs.ln1.params.variance_c = 1.0;
    configs.ln1.params.expected_input_level = 7;
    configs.ln1.params.expected_output_level = 5;
    configs.ln1.params.expected_no_norm_level = 10;
    configs.ln1.params.expected_inv_sqrt_var_level = 8;
    configs.ln1.backend = make_real_layernorm_backend(env);

    configs.f4.fc1_params = {batch, hidden, row, ffn, hidden, k_step};
    configs.f4.fc2_params = {batch, ffn, row, hidden, ffn, k_step};
    configs.f4.gelu_callbacks = make_real_gelu_callbacks(env);
    configs.f4.drop_non_cached_intermediates = true;
    configs.f4.ln1_input_level=7;
    configs.f4.fc1_complex_level=6;
    configs.f4.fc1_real_level=5;
    configs.f4.gelu_params.expected_input_level=5;

    configs.ln2_bts.ln2_config.params.row = row;
    configs.ln2_bts.ln2_config.params.col = hidden;
    configs.ln2_bts.ln2_config.params.batch_size = batch;
    configs.ln2_bts.ln2_config.params.k = k_step;
    configs.ln2_bts.ln2_config.params.gamma = assets.ln2_gamma;
    configs.ln2_bts.ln2_config.params.beta = assets.ln2_beta;
    configs.ln2_bts.ln2_config.params.variance_c = 8.0;
    configs.ln2_bts.ln2_config.params.expected_no_norm_level = 6;
    configs.ln2_bts.ln2_config.params.expected_inv_sqrt_var_level = 4;
    configs.ln2_bts.ln2_config.backend = make_real_layernorm_backend(env);
    configs.ln2_bts.bts_callbacks = real_bootstrap_callbacks;

    configs.classifier_forward.callbacks = real_bootstrap_callbacks;
    configs.classifier_forward.probability_backend=[&env](FHEAdapterContext &adapter,const FHEBlockTensor &logits,int level,const std::string &name) {
        auto out=make_like_metadata(logits,name);auto x=logits.cts.at(0);PhantomCiphertext swapped;
        env.evaluator.rotate_vector(x,env.encoder.slot_count()/2,env.galois_keys,swapped);env.evaluator.sub_inplace(x,swapped);
        auto half=encode_real_constant_like(adapter,x,0.5);x=multiply_plain_and_rescale(adapter,x,half,name+"_half_difference");
        PhantomCiphertext gate;apply_function(env,"tanh",gate,x);
        half=encode_real_constant_like(adapter,gate,0.5);gate=multiply_plain_and_rescale(adapter,gate,half,name+"_half_tanh");
        half=encode_real_constant_like(adapter,gate,0.5);env.evaluator.add_plain_inplace(gate,half);gate=polynomial_scalar(env,gate,1.,env.scale);out.cts.push_back(std::move(gate));
        refresh_tensor_metadata(env.context,env.level_map,out);align_down_to(env.context,env.level_map,out,level,out.name);return out;
    };

    configs.classifier_forward.c1_params = {batch, hidden, row, hidden, hidden, k_step};
    configs.classifier_forward.c2_params = {batch, hidden, 1, 2, hidden, k_step};
    if (!assets.W_Q.empty()) {
        configs.classifier_forward.bias_c1 = matrix_to_vector(
            load_matrix("src/data/pooler/parms/pooler_layer_bias.bin"), "pooler bias", hidden);
        configs.classifier_forward.bias_c2 = matrix_to_vector(
            load_matrix("src/data/classifier/parms/classifier_layer_bias.bin"), "classifier bias", 2);
    }

    configs.classifier_backward.callbacks = real_bootstrap_callbacks;
    configs.classifier_backward.cpu_params.batch_size = batch;
    configs.classifier_backward.cpu_params.row_X = row;
    configs.classifier_backward.cpu_params.col_W = hidden;
    configs.classifier_backward.cpu_params.pre_blocks = hidden / row;
    configs.classifier_backward.cpu_params.column_blocks =
        row * hidden * batch / static_cast<int>(env.encoder.slot_count());
    configs.classifier_backward.write_cache = false;

    configs.transformer_backward.callbacks = real_bootstrap_callbacks;
    enable_cpu_style_transformer_backward_ffn_ln(configs.transformer_backward, batch, row, hidden, ffn);
    enable_cpu_style_transformer_backward_attention_head(configs.transformer_backward,
                                                         batch,
                                                         row,
                                                         head_dim,
                                                         num_heads);
    enable_cpu_style_transformer_backward_lora_and_x(configs.transformer_backward,
                                                     batch,
                                                     row,
                                                     hidden,
                                                     rank);
    configs.transformer_backward.ln1_backward_params.gamma = assets.ln1_gamma;
    configs.transformer_backward.ln2_backward_params.gamma = assets.ln2_gamma;
    configs.transformer_backward.write_cache = false;
    configs.transformer_backward.release_forward_cache_after_read = true;
    configs.transformer_backward.release_intermediates_after_use = true;
    configs.transformer_backward.gradient_b_input_level = 11;
    configs.transformer_backward.lora_gradient_params.batch_size = batch;
    configs.transformer_backward.lora_gradient_params.row = row;
    configs.transformer_backward.lora_gradient_params.col = hidden;
    configs.transformer_backward.lora_gradient_params.rank = rank;
    configs.transformer_backward.lora_gradient_params.k = k_step;

    configs.adamw.callbacks = real_bootstrap_callbacks;
    configs.adamw.gradient_bts = real_bootstrap_callbacks.ordinary_bts;

    configs.adamw.inv_sqrt_backend = make_real_adamw_inv_sqrt_backend(env);
    configs.adamw.inv_sqrt_output_level = 7;
    configs.adamw.max_dividing_factor = 1.0;
    configs.adamw.learning_rate = 1e-3;
    configs.adamw.epsilon = 1e-4;

    configs.adamw.keep_intermediate_tensors = false;
    configs.adamw.keep_moment_tensors = true;
    return configs;
}

struct AbsoluteError {
    double maximum=0;
    long double sum=0;
    size_t count=0;
    void add(double value,double expected) {
        const double error=std::abs(value-expected);
        if(!std::isfinite(error))throw std::runtime_error("Nonfinite absolute error");
        maximum=std::max(maximum,error);sum+=error;++count;
    }
    void merge(const AbsoluteError &other) {
        maximum=std::max(maximum,other.maximum);sum+=other.sum;count+=other.count;
    }
    double mean()const{return count ? static_cast<double>(sum/count) : 0;}
};
void emit(const std::string &name,std::optional<AbsoluteError> error,std::optional<double> seconds={}) {
    phantom::execution::Scope validation(phantom::execution::Phase::Validation);
    std::cout<<name<<',';
    if(error)std::cout<<std::setprecision(12)<<error->mean();
    std::cout<<',';
    if(error)std::cout<<std::setprecision(12)<<error->maximum;
    std::cout<<',';
    if(seconds)std::cout<<std::setprecision(8)<<*seconds;
    std::cout<<'\n'<<std::flush;
}
AbsoluteError classifier_error(Environment &env,const FHEBlockTensor &tensor,const std::string &name) {
    phantom::execution::Scope validation(phantom::execution::Phase::Validation);
    const auto expected=load_matrix("reference/"+name+".bin").front();
    auto ct=tensor.cts.at(0);const auto actual=decrypt_real(env,ct);
    AbsoluteError error;
    for(size_t i=0;i<actual.size();++i){
        if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite classifier output");
        error.add(actual[i],expected.at(i/(actual.size()/2)));
    }
    return error;
}
std::string parameter_name(int matrix) {
    if(matrix>=12)return matrix==12 ? "Wc1" : "Wc2";
    return "layer_"+std::to_string(matrix/6)+"_"+(matrix%6<3 ? "A_" : "B_")+std::string(1,"QKV"[matrix%3]);
}
AbsoluteError parameter_error(Environment &env,const CompactOptimizerLayout &layout,int matrix,const FHEBlockTensor &tensor) {
    phantom::execution::Scope validation(phantom::execution::Phase::Validation);
    const auto expected=load_matrix("reference/updated_"+parameter_name(matrix)+".bin");
    const int features=layout.slots/(layout.batch*layout.row),tiles=layout.hidden/features;
    AbsoluteError error;
    for(size_t tile=0;tile<tensor.cts.size();++tile){
        auto ct=tensor.cts[tile];const auto actual=decrypt_complex(env,ct);
        for(int slot=0;slot<layout.slots;++slot)
            for(int component=0;component<(tensor.is_complex ? 2 : 1);++component){
                int row=0,col=0;
                if(matrix<12){
                    const bool b=matrix%6>=3;
                    const int real_tile=b ? 2*tile+component : tile;
                    const int feature=(real_tile%tiles)*features+slot/(layout.batch*layout.row);
                    const int rank=real_tile/tiles;
                    row=b ? rank : feature;col=b ? feature : rank;
                }else if(matrix==12){
                    row=(tile%tiles)*features+slot/(layout.batch*layout.row);
                    col=(tile/tiles)*layout.row+(slot/layout.batch)%layout.row;
                }else{
                    row=(slot%(layout.slots/2))/layout.batch;col=slot/(layout.slots/2);
                }
                if(row>=static_cast<int>(expected.size()))continue;
                const double value=component ? actual[slot].imag() : actual[slot].real();
                if(!std::isfinite(value))throw std::runtime_error("Nonfinite parameter output");
                error.add(value,expected.at(row).at(col));
            }
    }
    const size_t parameters=expected.size()*expected.front().size();
    error.sum*=static_cast<long double>(parameters)/error.count;
    error.count=parameters;
    return error;
}
void forward_layer(Environment &env,LayerInputs &inputs,const PipelineConfigs &config,LayerResult &result,int layer) {
    const std::string name="layer_"+std::to_string(layer);
    result.f1=forward_block_f1_qkv_lora(env.adapter,inputs.X,inputs.W_Q,inputs.W_K,inputs.W_V,
        inputs.A_Q,inputs.B_Q,inputs.A_K,inputs.B_K,inputs.A_V,inputs.B_V,config.f1,result.forward_cache,name+"_qkv");
    result.f2=forward_block_f2_attention(env.adapter,result.f1.Q_complex,result.f1.K_complex,result.f1.V_complex,
        config.f2,result.forward_cache,name+"_attention");
    result.f3=forward_block_f3_attention_projection(env.adapter,result.f2.H_merged,inputs.X,inputs.W_o,
        config.f3,result.forward_cache,name+"_projection");
    result.f1=ForwardF1Result{};result.f2=ForwardF2Result{};
    inputs.backward_inputs.X_for_lora_grad=std::move(inputs.X);
    result.ln1=forward_block_f3_layernorm1(env.adapter,result.f3.X_after_attn_residual,config.ln1,
        result.forward_cache,name+"_ln1");
    result.ln1.no_norm=FHEBlockTensor{};result.ln1.inv_sqrt_var=FHEBlockTensor{};
    auto ln1_low=result.ln1.output;
    align_down_to(env.context,env.level_map,ln1_low,0,ln1_low.name);
    auto ln1_input=ordinary_bts(env.adapter,ln1_low,config.f1.callbacks,
        "ln1_input",7,name+"_ln1_refresh");
    result.f4=forward_block_f4_ffn(env.adapter,ln1_input,ln1_input,inputs.W_fc1,inputs.W_fc2,
        config.f4,result.forward_cache,name+"_ffn");
    result.ln2_bts=forward_block_f4_layernorm2_bts(env.adapter,result.f4.X_after_ffn_residual,
        config.ln2_bts,result.forward_cache,name+"_ln2");
    result.f3=ForwardF3Result{};result.ln1=LayerNormResult{};result.f4=ForwardF4Result{};
    result.ln2_bts.ln2=LayerNormResult{};
}
void run_experiment() {
    constexpr int batch=16,row=128,hidden=768,rank=2,ffn=3072,heads=12,head_dim=64,k=1;
    Environment env;
    struct CacheLifetime {~CacheLifetime(){phantom::execution::clear_encoding_cache();}} cache_lifetime;
    phantom::execution::active_timer->change(phantom::execution::Phase::Compute);
    auto layout=make_compact_optimizer_layout(env.encoder.slot_count(),batch,row,hidden,rank);
    CompactOptimizerState state;
    CompactPackingBuffer theta(layout),gradient(layout);
    std::array<LayerInputs,2> inputs;
    std::array<LayerResult,2> layers;
    std::array<PipelineConfigs,2> configs;
    for(int layer=0;layer<2;++layer){
        {
        phantom::execution::Scope setup(phantom::execution::Phase::Setup);
        auto assets=load_real_layer_assets(layer,batch,row,hidden,ffn);
        inputs[layer]=make_real_shape_layer_inputs(env,assets,"layer_"+std::to_string(layer),batch,row,hidden,rank,k,layer==0);
        configs[layer]=make_real_shape_configs(env,assets,batch,row,hidden,ffn,rank,heads,head_dim,k);
        if(layer)inputs[layer].X=std::move(layers[0].ln2_bts.X_after_ln2);
        }
        forward_layer(env,inputs[layer],configs[layer],layers[layer],layer);
    }

    FHEBlockTensor wc1,wc2,target;
    {
    phantom::execution::Scope setup(phantom::execution::Phase::Setup);
    wc1=class1_packing_batch_encrypt(env,transpose_matrix(load_matrix("pooler/parms/pooler_layer_weight.bin")),
        batch,row,hidden,hidden,"Wc1");
    wc2=class2_packing_batch_encrypt(env,transpose_matrix(load_matrix("classifier/parms/classifier_layer_weight.bin")),
        batch,hidden,2,"Wc2");
    std::vector<double> labels(env.encoder.slot_count(),0);
    std::fill(labels.begin(),labels.begin()+labels.size()/2,1);
    target=encrypt_real_tensor(env,labels,"target",{1,2},PackingType::InterleavedRow);
    align_down_to(env.context, env.level_map, target, 3, target.name);
    }
    FHEForwardCache classifier_cache;
    auto fwd=classifier_forward(env.adapter,layers[1].ln2_bts.X_after_ln2,wc1,wc2,target,
        configs[1].classifier_forward,classifier_cache,"classifier_forward");
    const auto logits_error=classifier_error(env,fwd.X1Wc2,"logits");
    const auto probabilities_error=classifier_error(env,fwd.y_pred,"probabilities");
    emit("logits",logits_error);emit("probabilities",probabilities_error);

    FHEBackwardCache classifier_backward_cache;
    auto bwd=classifier_backward(env.adapter,fwd.y_lr,fwd.tanh_classifier,wc2,wc1,
        layers[1].ln2_bts.X_after_ln2,configs[1].classifier_backward,classifier_backward_cache,"classifier_backward");

    fwd=ClassifierForwardResult{};classifier_cache=FHEForwardCache{};
    layers[1].ln2_bts.X_after_ln2=FHEBlockTensor{};target=FHEBlockTensor{};
    FHEBlockTensor upstream=std::move(bwd.gradient_x2);
    for(int layer=1;layer>=0;--layer){
        TransformerBackwardInputs backward_inputs=inputs[layer].backward_inputs;
        fill_transformer_backward_plaintexts(backward_inputs,inputs[layer]);
        layers[layer].backward=transformer_backward(env.adapter,upstream,layers[layer].forward_cache,
            backward_inputs,configs[layer].transformer_backward,layers[layer].backward_cache,"layer_"+std::to_string(layer)+"_backward");
        const auto parameters=pipeline_lora_parameters(inputs[layer]);
        const auto gradients=pipeline_lora_gradients(layers[layer].backward);
        for(int matrix=0;matrix<6;++matrix){
            compact_pack_append(env.adapter,layout,theta,layer*6+matrix,*parameters[matrix],false);
            compact_pack_append(env.adapter,layout,gradient,layer*6+matrix,*gradients[matrix],true);
        }
        upstream=std::move(layers[layer].backward.gradient_X);
        layers[layer]=LayerResult{};inputs[layer]=LayerInputs{};
    }
    upstream=FHEBlockTensor{};
    auto g1=classifier_w_c1_gradient_as_parameter_packing(bwd.grad_Wc1_packed,wc1);
    auto g2=classifier_w_c2_gradient_as_parameter_packing(bwd.grad_Wc2_packed,wc2);
    compact_pack_append(env.adapter,layout,theta,12,wc1,false);
    compact_pack_append(env.adapter,layout,theta,13,wc2,false);
    compact_pack_append(env.adapter,layout,gradient,12,g1,true);
    compact_pack_append(env.adapter,layout,gradient,13,g2,true);
    g1=FHEBlockTensor{};g2=FHEBlockTensor{};bwd=ClassifierBackwardResult{};
    wc1=FHEBlockTensor{};wc2=FHEBlockTensor{};
    state.theta=compact_pack_finish(env.adapter,layout,std::move(theta),configs[1].adamw,"theta");
    auto compact_gradient=compact_pack_finish(env.adapter,layout,std::move(gradient),configs[1].adamw,"gradient");
    if(state.theta.cts.size()!=10 || compact_gradient.cts.size()!=10)
        throw std::runtime_error("Expected 10 compact ciphertexts");
    env.level_map=FHELevelMap::from_context_top_native_depth(env.context,13,14);
    env.adapter.level_map=env.level_map;
    compact_optimizer_step(env.adapter,layout,state,compact_gradient,configs[1].adamw);
    compact_gradient=FHEBlockTensor{};

    AbsoluteError combined;
    std::array<FHEBlockTensor,14> next_parameters;
    for(int matrix=0;matrix<14;++matrix){
        next_parameters[matrix]=pipeline_unpack_parameter(env.adapter,layout,state,matrix,configs[1].adamw);
        const auto error=parameter_error(env,layout,matrix,next_parameters[matrix]);
        emit("updated_"+parameter_name(matrix),error);combined.merge(error);
    }
    emit("updated_parameters",combined);
    if(omp_get_max_threads()!=1 || !fhe_stream_parallel_pool().empty())
        throw std::runtime_error("Expected a single execution stream");
    phantom::execution::active_timer->change(phantom::execution::Phase::Cleanup);
}
}
int main() {
    try {
        std::cout<<"metric,mean_absolute_error,max_absolute_error,seconds\n";
        omp_set_dynamic(0);omp_set_num_threads(1);
        phantom::execution::Timer timer;
        run_experiment();
        timer.finish();
        std::cerr<<"batch_size=16 streams=1 encoding_cache_hits="<<phantom::execution::encoding_cache_hits()
                 <<" encoding_cache_misses="<<phantom::execution::encoding_cache_misses()<<'\n';
        using phantom::execution::Phase;
        emit("online_compute",{},timer.seconds(Phase::Compute));
        return 0;
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
