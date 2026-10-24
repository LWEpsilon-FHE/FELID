#pragma once
#include <fstream>

#include "context.cuh"

#include "plaintext.h"
#include "ciphertext.h"
#include "prng.cuh"

class PhantomSecretKey;

class PhantomPublicKey;

class PhantomKSwitchKey;

class PhantomRelinKey;

class PhantomGaloisKey;

class PhantomPublicKey {

    friend class PhantomSecretKey;

private:

    bool gen_flag_ = false;
    PhantomCiphertext pk_;
    phantom::util::cuda_auto_ptr<uint8_t> prng_seed_a_;  

    void encrypt_zero_asymmetric_internal_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                                   size_t chain_index,
                                                   bool is_ntt_form, const cudaStream_t &stream) const;

public:

    void encrypt_zero_asymmetric_internal(const PhantomContext &context, PhantomCiphertext &cipher,
                                          size_t chain_index, const cudaStream_t &stream) const;

public:

    PhantomPublicKey() = default;

    PhantomPublicKey(const PhantomPublicKey &) = delete;

    PhantomPublicKey &operator=(const PhantomPublicKey &) = delete;

    PhantomPublicKey(PhantomPublicKey &&) = default;

    PhantomPublicKey &operator=(PhantomPublicKey &&) = default;

    ~PhantomPublicKey() = default;

    void encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain, PhantomCiphertext &cipher,
                            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    inline PhantomCiphertext encrypt_asymmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                                                const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        PhantomCiphertext cipher;
        encrypt_asymmetric(context, plain, cipher, stream_wrapper);
        return cipher;
    }
};

class PhantomRelinKey {

    friend class PhantomSecretKey;

private:

    bool gen_flag_ = false;

    std::vector<phantom::util::cuda_auto_ptr<uint64_t>> public_keys_;
    phantom::util::cuda_auto_ptr<uint64_t *> public_keys_ptr_;

public:

    PhantomRelinKey() = default;

    PhantomRelinKey(const PhantomRelinKey &) = delete;

    PhantomRelinKey &operator=(const PhantomRelinKey &) = delete;

    PhantomRelinKey(PhantomRelinKey &&) = default;

    PhantomRelinKey &operator=(PhantomRelinKey &&) = default;

    ~PhantomRelinKey() = default;

    [[nodiscard]] inline auto public_keys_ptr() const {
        return public_keys_ptr_.get();
    }

    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;

    [[nodiscard]] inline auto &parms_id() const {
        return parms_id_;
    }
};

class PhantomKSwitchKey {

    friend class PhantomSecretKey;

private:

    bool gen_flag_ = false;

    std::vector<phantom::util::cuda_auto_ptr<uint64_t>> public_keys_;
    phantom::util::cuda_auto_ptr<uint64_t *> public_keys_ptr_;

public:

    PhantomKSwitchKey() = default;

    PhantomKSwitchKey(const PhantomKSwitchKey &) = delete;

    PhantomKSwitchKey &operator=(const PhantomKSwitchKey &) = delete;

    PhantomKSwitchKey(PhantomKSwitchKey &&) = default;

    PhantomKSwitchKey &operator=(PhantomKSwitchKey &&) = default;

    ~PhantomKSwitchKey() = default;

    [[nodiscard]] inline auto public_keys_ptr() const {
        return public_keys_ptr_.get();
    }

    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;

    [[nodiscard]] inline auto &parms_id() const {
        return parms_id_;
    }
};

class PhantomGaloisKey {

    friend class PhantomSecretKey;

private:

    bool gen_flag_ = false;
    phantom::parms_id_type parms_id_ = phantom::parms_id_zero;
    std::vector<PhantomRelinKey> relin_keys_;

public:

    PhantomGaloisKey() = default;

    PhantomGaloisKey(const PhantomGaloisKey &) = delete;

    PhantomGaloisKey &operator=(const PhantomGaloisKey &) = delete;

    PhantomGaloisKey(PhantomGaloisKey &&) = default;

    PhantomGaloisKey &operator=(PhantomGaloisKey &&) = default;

    ~PhantomGaloisKey() = default;

    [[nodiscard]] auto &get_relin_keys(size_t index) const {
        return relin_keys_.at(index);
    }
};

class PhantomSecretKey {

private:

    bool gen_flag_ = false;
    uint64_t chain_index_ = 0;
    uint64_t sk_max_power_ = 0;  
    uint64_t poly_modulus_degree_ = 0;
    uint64_t coeff_modulus_size_ = 0;

    phantom::util::cuda_auto_ptr<uint64_t> data_rns_;
    phantom::util::cuda_auto_ptr<uint64_t> secret_key_array_;  

    void compute_secret_key_array(const PhantomContext &context, size_t max_power, const cudaStream_t &stream);

    void gen_secretkey(const PhantomContext &context, const cudaStream_t &stream);

    void gen_secretkey_with_zero(const PhantomContext &context, const int zero_threshold, const cudaStream_t &stream);

    void load_secret_key_to_small(const PhantomContext &context, const std::vector<std::uint64_t> &src_array_data, const size_t small_degree, const cudaStream_t &stream);

    void
    bfv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                const cudaStream_t &stream);

    void
    ckks_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                 const cudaStream_t &stream);
     
    void
    ckks_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination, const size_t small_degree,
                 const cudaStream_t &stream);

    void
    bgv_decrypt(const PhantomContext &context, const PhantomCiphertext &encrypted, PhantomPlaintext &destination,
                const cudaStream_t &stream);

public:

    explicit inline PhantomSecretKey(const PhantomContext &context) {
        gen_secretkey(context, phantom::util::global_variables::default_stream->get_stream());
    }

    explicit inline PhantomSecretKey(const PhantomContext &context, const int threshold) {
        gen_secretkey_with_zero(context, threshold, phantom::util::global_variables::default_stream->get_stream());
    }

    void compute_ntt(const PhantomContext &context, const cudaStream_t &stream);

    explicit inline PhantomSecretKey(const PhantomContext &context, const std::vector<std::uint64_t> &src_array_data, const size_t small_degree) {
         
        load_secret_key_to_small(context, src_array_data, small_degree, phantom::util::global_variables::default_stream->get_stream());
        cudaDeviceSynchronize();
        compute_ntt(context, phantom::util::global_variables::default_stream->get_stream());
        cudaDeviceSynchronize();
    }

    void generate_one_kswitch_key(const PhantomContext &context, uint64_t *new_key, PhantomRelinKey &relin_key,
                                  const cudaStream_t &stream) const;

    void generate_one_kswitch_key(const PhantomContext &context, uint64_t *new_key,
                                                PhantomKSwitchKey &kswitch_key, const PhantomSecretKey &target_sk, const cudaStream_t &stream) const;
    
    PhantomSecretKey(const PhantomSecretKey &) = delete;

    PhantomSecretKey &operator=(const PhantomSecretKey &) = delete;

    PhantomSecretKey(PhantomSecretKey &&) = default;

    PhantomSecretKey &operator=(PhantomSecretKey &&) = default;

    ~PhantomSecretKey() = default;

    [[nodiscard]] PhantomPublicKey gen_publickey(const PhantomContext &context) const;

    [[nodiscard]] PhantomRelinKey gen_relinkey(const PhantomContext &context);

    [[nodiscard]] PhantomKSwitchKey gen_kswitchkey(const PhantomContext &context, const PhantomSecretKey &target_sk);

    [[nodiscard]] PhantomGaloisKey create_galois_keys(const PhantomContext &context) const;

    [[nodiscard]] PhantomGaloisKey create_galois_keys_from_elts(PhantomContext &context,const std::vector<uint32_t> &elts) const;

    [[nodiscard]] PhantomGaloisKey create_galois_keys_from_steps(PhantomContext &context, const std::vector<int> &steps) const; 

    void encrypt_zero_symmetric(const PhantomContext &context, PhantomCiphertext &cipher, const uint8_t *prng_seed_a,
                                size_t chain_index, bool is_ntt_form, const cudaStream_t &stream) const;

    void
    encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain, PhantomCiphertext &cipher,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const;

    void decrypt(const PhantomContext &context, const PhantomCiphertext &cipher, PhantomPlaintext &plain,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    void decrypt(const PhantomContext &context, const PhantomCiphertext &cipher, PhantomPlaintext &plain, const size_t small_degree,
                 const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    [[nodiscard]] inline PhantomCiphertext
    encrypt_symmetric(const PhantomContext &context, const PhantomPlaintext &plain,
                      const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) const {
        PhantomCiphertext cipher;
        encrypt_symmetric(context, plain, cipher, stream_wrapper);
        return cipher;
    }

    [[nodiscard]] inline PhantomPlaintext
    decrypt(const PhantomContext &context, const PhantomCiphertext &cipher,
            const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream) {
        PhantomPlaintext plain;
        decrypt(context, cipher, plain, stream_wrapper);
        return plain;
    }

    [[nodiscard]] int invariant_noise_budget(const PhantomContext &context, const PhantomCiphertext &cipher,
                                             const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    inline void load_secret_key(const PhantomContext &context, std::ifstream &sk_in) {
      std::string line;
      size_t total_line_count = 0;
    
      auto new_sk_array_data = new uint64_t[context.coeff_mod_size_ * context.poly_degree_];
    
      while (std::getline(sk_in, line)) {
        uint64_t value = std::stoull(line);
        new_sk_array_data[total_line_count] = value;
        total_line_count++;
      }
    
      if (total_line_count != context.coeff_mod_size_ * context.poly_degree_) {
        throw std::invalid_argument("Invalid secret key input.");
      }
    
      cudaMemcpy(secret_key_array_.get(), new_sk_array_data, context.coeff_mod_size_ * context.poly_degree_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
      delete[] new_sk_array_data;
    }

    [[nodiscard]] inline auto secret_key_array() const {
        return secret_key_array_.get();
    }
};
