#pragma once

#include <memory>
#include <string>

#include "../Bootstrapper.cuh"
#include "RbootSlimModularReducer.cuh"
#include "RbootSparseOps.cuh"

namespace rboot_slim {

class RbootSlimBootstrapper : public Bootstrapper {
public:
    std::unique_ptr<RbootSparseOps> sparse_ops;
    RbootSlimModularReducer *rboot_mod_reducer = nullptr;

    RbootSlimBootstrapper(long loge,
                          long logn,
                          long logNh,
                          long total_level,
                          double final_scale,
                          long boundary_K,
                          long sin_cos_deg,
                          long double_angle_iterations,
                          long inverse_deg,
                          moai::CKKSEvaluator *ckks,
                          PhantomCKKSEncoder *raw_encoder,
                          std::string arcsin_coeff_path = {});

    ~RbootSlimBootstrapper();

    void prepare_mod_polynomial(bool include_gelu_algo6 = false,
                                bool include_tanh = false,
                                bool include_exp = false);

    void rboot_bsgs_linear_transform(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn,
        const std::vector<std::vector<std::complex<double>>> &fftcoeff,
        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);
    void rboot_rotated_bsgs_linear_transform(
        PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn,
        const std::vector<std::vector<std::complex<double>>> &fftcoeff,
        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

    void slim_sfl_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_sfl_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_sflinv_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_sflinv_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void slim_coefftoslot_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_slottocoeff(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_coefftoslot(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_slottocoeff_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void ScaleDown(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

    void slim_bootstrap_sparse_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_full_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_gelu(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_gelu_algo6(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_tanh(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_exp(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
    void slim_bootstrap_inplace(PhantomCiphertext &cipher);
};

}
