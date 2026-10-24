#pragma once

#include <NTL/RR.h>

#include <string>
#include <vector>

#include "source/bootstrapping/common/PolyUpdate.cuh"
#include "source/bootstrapping/common/Polynomial.cuh"
#include "source/ckks_evaluator_parallel.cuh"
#include "RbootSparseOps.cuh"

namespace rboot_slim {

class RbootSlimModularReducer {
public:
    long boundary_K;
    double log_width;
    long deg;
    long num_double_formula;
    double inverse_log_width;
    long inverse_deg;

    moai::CKKSEvaluator *ckks = nullptr;
    RbootSparseOps *ops = nullptr;
    std::string arcsin_coeff_path;
    std::string gelu_even_coeff_path;
    std::string gelu_algo6_even_coeff_path;
    std::string tanh_coeff_path;
    std::string exp_coeff_path;

    boot::Polynomial sin_polynomial;
    boot::Polynomial cos_polynomial;
    boot::Polynomial inverse_sin_polynomial_v1;
    boot::Polynomial inverse_sin_polynomial_v1_original;
    boot::Polynomial gelu_even_polynomial;
    boot::Polynomial tanh_polynomial;
    boot::Polynomial exp_polynomial;

    std::vector<NTL::RR> arcsin_decomp_coeff;
    std::vector<NTL::RR> arcsin_decomp_coeff_original;
    minicomp::Tree arcsin_tree;
    std::vector<NTL::RR> gelu_algo6_even_decomp_coeff;
    std::vector<NTL::RR> gelu_algo6_even_decomp_coeff_original;
    minicomp::Tree gelu_algo6_even_tree;

    double abs_lift = 0.125;
    double scale_for_eval = 4.0 * 0.9;
    double scale_for_boost_relu_range = 2.0;
    double gelu_target_input_scale = 1.0;
    double dyn_scale_for_turn_back_q;
    bool after_cnn_first_boot = false;

    RbootSlimModularReducer(long boundary_K,
                            double log_width,
                            long deg,
                            long num_double_formula,
                            long inverse_deg,
                            moai::CKKSEvaluator *ckks,
                            RbootSparseOps *ops,
                            std::string arcsin_coeff_path = {});

    void double_angle_formula(PhantomCiphertext &cipher);
    void double_angle_formula_scaled(PhantomCiphertext &cipher, double scale_coeff);
    void generate_sin_cos_polynomial();
    void generate_inverse_sine_polynomial();
    void generate_gelu_even_polynomial();
    void generate_gelu_algo6_even_decomposition();
    void generate_tanh_polynomial();
    void generate_exp_polynomial();
    void scaling_for_turn_back_q();
    void modular_reduction_split(PhantomCiphertext &odd_rtn,
                                 PhantomCiphertext &even_rtn,
                                 PhantomCiphertext &cipher);
    void modular_reduction(PhantomCiphertext &rtn, PhantomCiphertext &cipher);
};

}
