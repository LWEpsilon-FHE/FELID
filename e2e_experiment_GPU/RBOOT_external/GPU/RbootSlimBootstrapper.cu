#include "RbootSlimBootstrapper.cuh"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "RbootSlimPolyEval.cuh"

using namespace std;
using namespace phantom;
using namespace moai;

namespace rboot_slim {
namespace {

}   

RbootSlimBootstrapper::RbootSlimBootstrapper(long loge,
                                             long logn,
                                             long logNh,
                                             long total_level,
                                             double final_scale,
                                             long boundary_K,
                                             long sin_cos_deg,
                                             long double_angle_iterations,
                                             long inverse_deg,
                                             CKKSEvaluator *ckks,
                                             PhantomCKKSEncoder *raw_encoder,
                                             std::string arcsin_coeff_path)
    : Bootstrapper(loge,
                   logn,
                   logNh,
                   total_level,
                   final_scale,
                   boundary_K,
                   sin_cos_deg,
                   double_angle_iterations,
                   inverse_deg,
                   ckks)
{
    sparse_ops = std::make_unique<RbootSparseOps>(ckks, raw_encoder);
    rboot_mod_reducer = new RbootSlimModularReducer(boundary_K,
                                                    static_cast<double>(loge),
                                                    sin_cos_deg,
                                                    double_angle_iterations,
                                                    inverse_deg,
                                                    ckks,
                                                    sparse_ops.get(),
                                                    std::move(arcsin_coeff_path));
}

RbootSlimBootstrapper::~RbootSlimBootstrapper()
{
    delete rboot_mod_reducer;
}

void RbootSlimBootstrapper::prepare_mod_polynomial(bool include_gelu_algo6, bool include_tanh, bool include_exp)
{
    rboot_mod_reducer->generate_sin_cos_polynomial();
    rboot_mod_reducer->generate_inverse_sine_polynomial();
    rboot_mod_reducer->generate_gelu_even_polynomial();
    if (include_tanh)
    {
        rboot_mod_reducer->generate_tanh_polynomial();
    }
    if (include_exp)
    {
        rboot_mod_reducer->generate_exp_polynomial();
    }
    if (include_gelu_algo6)
    {
        rboot_mod_reducer->generate_gelu_algo6_even_decomposition();
    }
}

void RbootSlimBootstrapper::rboot_bsgs_linear_transform(
    PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn,
    const vector<vector<complex<double>>> &fftcoeff, const cuda_stream_wrapper &stream_wrapper)
{
    int gs1 = giantstep(2 * totlen + 1);
    int basicstart1 = -totlen + gs1 * floor((totlen + 0.0) / (gs1 + 0.0));
    int giantfirst1 = -floor((totlen + 0.0) / (gs1 + 0.0));
    int giantlast1 = floor((2 * totlen + 0.0) / (gs1 + 0.0)) + giantfirst1;

    vector<PhantomCiphertext> babyct(gs1, PhantomCiphertext());
    PhantomCiphertext giantct, tmpct, tmptmpct;
    bool giantbool = false, tmpctbool = false;
    vector<complex<double>> rotatedcoeff;
    rotatedcoeff.reserve(Nh);

    for (int i = basicstart1; i < basicstart1 + gs1; i++)
    {
        if (i == 0)
        {
            babyct[i - basicstart1] = deep_copy_ct(cipher, *(ckks->context), stream_wrapper);
        }
        else
        {
            ckks->evaluator.rotate_vector(cipher, (Nh + i * basicstep) % Nh, *(ckks->galois_keys), babyct[i - basicstart1], stream_wrapper);
        }
    }

    for (int i = giantfirst1; i <= giantlast1; i++)
    {
        giantbool = false;
        const int j_end = (i != giantlast1) ? (basicstart1 + gs1 - 1) : (totlen - i * gs1);
        for (int j = basicstart1; j <= j_end; j++)
        {
            rotation(coeff_logn, Nh, (-i) * gs1 * basicstep, fftcoeff[(i * gs1 + j) + totlen], rotatedcoeff);
            sparse_ops->multiply_vector_reduced_error(babyct[j - basicstart1], rotatedcoeff, tmptmpct, stream_wrapper);
            if (!giantbool)
            {
                giantct = deep_copy_ct(tmptmpct, *(ckks->context), stream_wrapper);
                giantbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(giantct, tmptmpct, stream_wrapper);
            }
        }

        if (i != 0)
        {
            ckks->evaluator.rotate_vector(giantct, (Nh + i * gs1 * basicstep) % Nh, *(ckks->galois_keys), tmptmpct, stream_wrapper);
            if (!tmpctbool)
            {
                tmpct = deep_copy_ct(tmptmpct, *(ckks->context), stream_wrapper);
                tmpctbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(tmpct, tmptmpct, stream_wrapper);
            }
        }
        else
        {
            if (!tmpctbool)
            {
                tmpct = deep_copy_ct(giantct, *(ckks->context), stream_wrapper);
                tmpctbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(tmpct, giantct, stream_wrapper);
            }
        }
    }
    rtncipher = deep_copy_ct(tmpct, *(ckks->context), stream_wrapper);
}

void RbootSlimBootstrapper::rboot_rotated_bsgs_linear_transform(
    PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn,
    const vector<vector<complex<double>>> &fftcoeff, const cuda_stream_wrapper &stream_wrapper)
{
    int gs2 = giantstep(totlen + 1);
    int giantlast2 = floor((totlen + 0.0) / (gs2 + 0.0));

    vector<PhantomCiphertext> babyct(gs2, PhantomCiphertext());
    PhantomCiphertext giantct, tmpct, tmptmpct;
    bool giantbool = false, tmpctbool = false;
    vector<complex<double>> rotatedcoeff;
    rotatedcoeff.reserve(Nh);

    for (int i = 0; i < gs2; i++)
    {
        if (i == 0)
        {
            babyct[i] = deep_copy_ct(cipher, *(ckks->context), stream_wrapper);
        }
        else
        {
            ckks->evaluator.rotate_vector(cipher, (Nh + i * basicstep) % Nh, *(ckks->galois_keys), babyct[i], stream_wrapper);
        }
    }

    for (int i = 0; i <= giantlast2; i++)
    {
        giantbool = false;
        const int j_end = (i != giantlast2) ? (gs2 - 1) : (totlen - i * gs2);
        for (int j = 0; j <= j_end; j++)
        {
            rotation(coeff_logn, Nh, (-i) * gs2 * basicstep, fftcoeff[i * gs2 + j], rotatedcoeff);
            sparse_ops->multiply_vector_reduced_error(babyct[j], rotatedcoeff, tmptmpct, stream_wrapper);
            if (!giantbool)
            {
                giantct = deep_copy_ct(tmptmpct, *(ckks->context), stream_wrapper);
                giantbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(giantct, tmptmpct, stream_wrapper);
            }
        }

        if (i != 0)
        {
            ckks->evaluator.rotate_vector(giantct, (Nh + i * gs2 * basicstep) % Nh, *(ckks->galois_keys), tmptmpct, stream_wrapper);
            if (!tmpctbool)
            {
                tmpct = deep_copy_ct(tmptmpct, *(ckks->context), stream_wrapper);
                tmpctbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(tmpct, tmptmpct, stream_wrapper);
            }
        }
        else
        {
            if (!tmpctbool)
            {
                tmpct = deep_copy_ct(giantct, *(ckks->context), stream_wrapper);
                tmpctbool = true;
            }
            else
            {
                sparse_ops->add_inplace_reduced_error(tmpct, giantct, stream_wrapper);
            }
        }
    }
    rtncipher = deep_copy_ct(tmpct, *(ckks->context), stream_wrapper);
}

void RbootSlimBootstrapper::slim_sfl_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    const int div_part3 = floor(logn / 3.0);
    const int div_part2 = floor((logn - div_part3) / 2.0);
    const int div_part1 = logn - div_part3 - div_part2;

    const int totlen1 = (1 << div_part1) - 1;
    const int totlen2 = (1 << div_part2) - 1;
    const int totlen3 = (1 << div_part3) - 1;

    const int basicstep1 = 1;
    const int basicstep2 = (1 << div_part1);
    const int basicstep3 = (1 << (div_part1 + div_part2));

    PhantomCiphertext tmpct;
    rboot_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn + 1, fftcoeff1[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct);

    PhantomCiphertext tmpct2;
    rboot_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn + 1, fftcoeff2[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct2);

    auto fftcoeff3_scaled = fftcoeff3[slot_index];
    for (auto &row : fftcoeff3_scaled)
    {
        for (auto &coeff : row)
        {
            coeff *= 2.0;
            coeff *= rboot_mod_reducer->scale_for_eval;
            coeff *= rboot_mod_reducer->scale_for_boost_relu_range;
        }
    }

    rboot_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn + 1, fftcoeff3_scaled);
    ckks->evaluator.rescale_to_next_inplace(rtncipher);
}

void RbootSlimBootstrapper::slim_sfl_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    const int div_part3 = floor(logn / 3.0);
    const int div_part2 = floor((logn - div_part3) / 2.0);
    const int div_part1 = logn - div_part3 - div_part2;

    const int totlen1 = (1 << div_part1) - 1;
    const int totlen2 = (1 << div_part2) - 1;
    const int totlen3 = (1 << div_part3) - 1;

    const int basicstep1 = 1;
    const int basicstep2 = (1 << div_part1);
    const int basicstep3 = (1 << (div_part1 + div_part2));

    PhantomCiphertext tmpct;
    rboot_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, fftcoeff1[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct);

    PhantomCiphertext tmpct2;
    rboot_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, fftcoeff2[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct2);

    auto fftcoeff3_scale = fftcoeff3[slot_index];
    for (auto &row : fftcoeff3_scale)
    {
        for (auto &coeff : row)
        {
            coeff *= rboot_mod_reducer->scale_for_eval;
            coeff *= rboot_mod_reducer->scale_for_boost_relu_range;
        }
    }

    rboot_rotated_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn, fftcoeff3_scale);
    ckks->evaluator.rescale_to_next_inplace(rtncipher);
}

void RbootSlimBootstrapper::slim_sflinv_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    const int div_part1 = floor(logn / 3.0);
    const int div_part2 = floor((logn - div_part1) / 2.0);
    const int div_part3 = logn - div_part1 - div_part2;

    const int totlen1 = (1 << div_part1) - 1;
    const int totlen2 = (1 << div_part2) - 1;
    const int totlen3 = (1 << div_part3) - 1;

    const int basicstep1 = (1 << (logn - div_part1));
    const int basicstep2 = (1 << (logn - div_part1 - div_part2));
    const int basicstep3 = 1;

    PhantomCiphertext tmpct;
    rboot_rotated_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, invfftcoeff1[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct);

    PhantomCiphertext tmpct2;
    rboot_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, invfftcoeff2[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct2);

    rboot_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn + 1, invfftcoeff3[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(rtncipher);
}

void RbootSlimBootstrapper::slim_sflinv_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    const int div_part1 = floor(logn / 3.0);
    const int div_part2 = floor((logn - div_part1) / 2.0);
    const int div_part3 = logn - div_part1 - div_part2;

    const int totlen1 = (1 << div_part1) - 1;
    const int totlen2 = (1 << div_part2) - 1;
    const int totlen3 = (1 << div_part3) - 1;

    const int basicstep1 = (1 << (logn - div_part1));
    const int basicstep2 = (1 << (logn - div_part1 - div_part2));
    const int basicstep3 = 1;

    PhantomCiphertext tmpct;
    rboot_rotated_bsgs_linear_transform(tmpct, cipher, totlen1, basicstep1, logn, invfftcoeff1[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct);

    PhantomCiphertext tmpct2;
    rboot_bsgs_linear_transform(tmpct2, tmpct, totlen2, basicstep2, logn, invfftcoeff2[slot_index]);
    ckks->evaluator.rescale_to_next_inplace(tmpct2);

    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    const auto curr_level = ckks->context->get_context_data(tmpct2.params_id()).chain_depth();
    const double mod_zero = static_cast<double>(modulus[0].value());
    const double curr_mod = static_cast<double>(modulus[curr_level].value());

    vector<vector<complex<double>>> invfftcoeff3_scale(2 * totlen3 + 1);
    for (int i = 0; i < 2 * totlen3 + 1; ++i)
    {
        invfftcoeff3_scale[i].resize(n);
    }

    for (int i = 0; i < 2 * totlen3 + 1; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            invfftcoeff3_scale[i][j] = invfftcoeff3[slot_index][i][j] * curr_mod * mod_zero * final_scale /
                                       (tmpct2.scale() * tmpct2.scale() * initial_scale);
        }
    }

    rboot_bsgs_linear_transform(rtncipher, tmpct2, totlen3, basicstep3, logn, invfftcoeff3_scale);
    ckks->evaluator.rescale_to_next_inplace(rtncipher);
}

void RbootSlimBootstrapper::slim_coefftoslot_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    PhantomCiphertext tmpct1;
    PhantomCiphertext tmpct4;
    slim_sflinv_full_3(tmpct1, cipher);
    ckks->evaluator.complex_conjugate(tmpct1, *(ckks->galois_keys), tmpct4);
    sparse_ops->add_reduced_error(tmpct1, tmpct4, rtncipher);
}

void RbootSlimBootstrapper::slim_slottocoeff(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    PhantomCiphertext tmpct;
    slim_sfl_3(tmpct, cipher);
    rtncipher = tmpct;
}

void RbootSlimBootstrapper::slim_coefftoslot(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    PhantomCiphertext tmpct;
    slim_sflinv_3(tmpct, cipher);
    rtncipher = tmpct;
}

void RbootSlimBootstrapper::slim_slottocoeff_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    PhantomCiphertext tmpct;
    slim_sfl_full_3(tmpct, cipher);
    rtncipher = tmpct;
}

void RbootSlimBootstrapper::ScaleDown(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    const auto curr_level = ckks->context->get_context_data(cipher.params_id()).chain_depth();
    const double mod_zero = static_cast<double>(modulus[0].value());
    const double curr_mod = static_cast<double>(modulus[curr_level].value());
    const double scale_up = curr_mod * mod_zero * final_scale / (cipher.scale() * cipher.scale() * initial_scale);

    vector<double> tmpvec(Nh, scale_up);
    PhantomPlaintext tmpplain;
    sparse_ops->encode_exact_vector(tmpvec, cipher.scale(), tmpplain);
    ckks->evaluator.mod_switch_to_inplace(tmpplain, cipher.params_id());
    ckks->evaluator.multiply_plain_inplace(rtncipher, tmpplain);
    ckks->evaluator.rescale_to_next_inplace(rtncipher);
}

void RbootSlimBootstrapper::slim_bootstrap_sparse_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    slim_slottocoeff(rtn, cipher);

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext rot;
    for (long i = logn; i < logNh; ++i)
    {
        ckks->evaluator.rotate_vector(rtn, (1 << i), *(ckks->galois_keys), rot);
        ckks->evaluator.add_inplace(rtn, rot);
    }

    PhantomCiphertext real_part;
    slim_coefftoslot(real_part, rtn);

    PhantomCiphertext conj;
    ckks->evaluator.complex_conjugate(real_part, *(ckks->galois_keys), conj);
    ckks->evaluator.add_inplace(real_part, conj);

    rboot_mod_reducer->modular_reduction(real_part, real_part);

    rtncipher = real_part;
    rtncipher.scale() = std::pow(2.0, 46);
}

void RbootSlimBootstrapper::slim_bootstrap_full_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    initial_scale = cipher.scale();

    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    slim_slottocoeff_full(rtn, cipher);

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext real_part;
    slim_coefftoslot_full(real_part, rtn);
    rboot_mod_reducer->modular_reduction(real_part, real_part);

    rtncipher = real_part;
    rtncipher.scale() = std::pow(2.0, 46);
}

void RbootSlimBootstrapper::slim_bootstrap(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    initial_scale = cipher.scale();
    if (logn == logNh)
    {
        slim_bootstrap_full_real_3(rtncipher, cipher);
    }
    else
    {
        slim_bootstrap_sparse_3(rtncipher, cipher);
    }
}

void RbootSlimBootstrapper::slim_bootstrap_inplace(PhantomCiphertext &cipher)
{
    PhantomCiphertext rtncipher;
    slim_bootstrap(rtncipher, cipher);
    cipher = rtncipher;
}

void RbootSlimBootstrapper::slim_bootstrap_gelu(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{

    initial_scale = cipher.scale();

    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    if (logn == logNh)
    {
        slim_slottocoeff_full(rtn, cipher);
    }
    else
    {
        slim_slottocoeff(rtn, cipher);
    }

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext real_part;
    if (logn == logNh)
    {
        slim_coefftoslot_full(real_part, rtn);
    }
    else
    {
        PhantomCiphertext rot;
        for (long i = logn; i < logNh; ++i)
        {
            ckks->evaluator.rotate_vector(rtn, (1 << i), *(ckks->galois_keys), rot);
            ckks->evaluator.add_inplace(rtn, rot);
        }

        slim_coefftoslot(real_part, rtn);

        PhantomCiphertext conj;
        ckks->evaluator.complex_conjugate(real_part, *(ckks->galois_keys), conj);
        ckks->evaluator.add_inplace(real_part, conj);
    }

    PhantomCiphertext odd_half;
    PhantomCiphertext even_half_abs;
    rboot_mod_reducer->modular_reduction_split(odd_half, even_half_abs, real_part);

    PhantomCiphertext gelu_even;
    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->gelu_even_polynomial, *sparse_ops, gelu_even, even_half_abs);

    const double target_input_scale = rboot_mod_reducer->gelu_target_input_scale;
    const long rounded_scale = static_cast<long>(std::llround(target_input_scale));
    if (std::abs(target_input_scale - static_cast<double>(rounded_scale)) > 1e-9 || rounded_scale < 1 ||
        (rounded_scale & (rounded_scale - 1)) != 0)
    {
        throw std::runtime_error("RBOOT_GELU_TARGET_INPUT_SCALE must be a positive power of two in the current GPU path.");
    }
    for (long k = 0; k < static_cast<long>(std::log2(static_cast<double>(rounded_scale))); ++k)
    {
        ckks->evaluator.double_inplace(odd_half);
    }

    sparse_ops->add_reduced_error(odd_half, gelu_even, rtncipher);
    rtncipher.scale() = std::pow(2.0, 46);
    
}

void RbootSlimBootstrapper::slim_bootstrap_tanh(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{

    initial_scale = cipher.scale();

    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    if (logn == logNh)
    {
        slim_slottocoeff_full(rtn, cipher);
    }
    else
    {
        slim_slottocoeff(rtn, cipher);
    }

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext real_part;
    if (logn == logNh)
    {
        slim_coefftoslot_full(real_part, rtn);
    }
    else
    {
        PhantomCiphertext rot;
        for (long i = logn; i < logNh; ++i)
        {
            ckks->evaluator.rotate_vector(rtn, (1 << i), *(ckks->galois_keys), rot);
            ckks->evaluator.add_inplace(rtn, rot);
        }

        slim_coefftoslot(real_part, rtn);

        PhantomCiphertext conj;
        ckks->evaluator.complex_conjugate(real_part, *(ckks->galois_keys), conj);
        ckks->evaluator.add_inplace(real_part, conj);
    }

    PhantomCiphertext sine_part;
    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->sin_polynomial, *sparse_ops, sine_part, real_part);
    for (int i = 0; i < rboot_mod_reducer->num_double_formula; ++i)
    {
        rboot_mod_reducer->double_angle_formula(sine_part);
    }

    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->tanh_polynomial, *sparse_ops, rtncipher, sine_part);
    rtncipher.scale() = std::pow(2.0, 46);
    
}

void RbootSlimBootstrapper::slim_bootstrap_exp(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{

    initial_scale = cipher.scale();

    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    if (logn == logNh)
    {
        slim_slottocoeff_full(rtn, cipher);
    }
    else
    {
        slim_slottocoeff(rtn, cipher);
    }

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext real_part;
    if (logn == logNh)
    {
        slim_coefftoslot_full(real_part, rtn);
    }
    else
    {
        PhantomCiphertext rot;
        for (long i = logn; i < logNh; ++i)
        {
            ckks->evaluator.rotate_vector(rtn, (1 << i), *(ckks->galois_keys), rot);
            ckks->evaluator.add_inplace(rtn, rot);
        }

        slim_coefftoslot(real_part, rtn);

        PhantomCiphertext conj;
        ckks->evaluator.complex_conjugate(real_part, *(ckks->galois_keys), conj);
        ckks->evaluator.add_inplace(real_part, conj);
    }

    PhantomCiphertext sine_part;
    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->sin_polynomial, *sparse_ops, sine_part, real_part);
    for (int i = 0; i < rboot_mod_reducer->num_double_formula; ++i)
    {
        rboot_mod_reducer->double_angle_formula(sine_part);
    }

    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->exp_polynomial, *sparse_ops, rtncipher, sine_part);
    rtncipher.scale() = std::pow(2.0, 46);
    
}

void RbootSlimBootstrapper::slim_bootstrap_gelu_algo6(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher)
{
    initial_scale = cipher.scale();

    auto curr_level = static_cast<int>(ckks->context->get_context_data(cipher.params_id()).chain_depth());
    if (curr_level < 3)
    {
        throw invalid_argument("curr_level is less than 3.");
    }
    while (curr_level > 3)
    {
        ckks->evaluator.mod_switch_to_next_inplace(cipher);
        --curr_level;
    }

    PhantomCiphertext rtn;
    if (logn == logNh)
    {
        slim_slottocoeff_full(rtn, cipher);
    }
    else
    {
        slim_slottocoeff(rtn, cipher);
    }

    modraise_inplace(rtn);
    const auto modulus = ckks->context->first_context_data().parms().coeff_modulus();
    rtn.scale() = static_cast<double>(modulus[0].value());

    PhantomCiphertext real_part;
    if (logn == logNh)
    {
        slim_coefftoslot_full(real_part, rtn);
    }
    else
    {
        PhantomCiphertext rot;
        for (long i = logn; i < logNh; ++i)
        {
            ckks->evaluator.rotate_vector(rtn, (1 << i), *(ckks->galois_keys), rot);
            ckks->evaluator.add_inplace(rtn, rot);
        }

        slim_coefftoslot(real_part, rtn);

        PhantomCiphertext conj;
        ckks->evaluator.complex_conjugate(real_part, *(ckks->galois_keys), conj);
        ckks->evaluator.add_inplace(real_part, conj);
    }

    rboot_mod_reducer->scaling_for_turn_back_q();

    PhantomCiphertext sin_tmp2;
    PhantomCiphertext cos_tmp2;
    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->sin_polynomial, *sparse_ops, cos_tmp2, real_part);
    for (int i = 0; i < rboot_mod_reducer->num_double_formula + 1; ++i)
    {
        rboot_mod_reducer->double_angle_formula(cos_tmp2);
        if (i == rboot_mod_reducer->num_double_formula - 1)
        {
            sin_tmp2 = cos_tmp2;
        }
    }

    PhantomCiphertext odd_half;
    homomorphic_poly_evaluation_sparse(rboot_mod_reducer->inverse_sin_polynomial_v1, *sparse_ops, odd_half, sin_tmp2);
    ckks->evaluator.mod_switch_to_next_inplace(odd_half);

    const long gelu_even_degree = std::getenv("RBOOT_GELU_ALGO6_EVEN_DEG") != nullptr
                                      ? std::strtol(std::getenv("RBOOT_GELU_ALGO6_EVEN_DEG"), nullptr, 10)
                                      : 127;
    PhantomCiphertext gelu_even_direct;
    eval_polynomial_integrate(*sparse_ops,
                              gelu_even_direct,
                              cos_tmp2,
                              gelu_even_degree,
                              rboot_mod_reducer->gelu_algo6_even_decomp_coeff,
                              rboot_mod_reducer->gelu_algo6_even_tree);

    sparse_ops->add_reduced_error(odd_half, gelu_even_direct, rtncipher);
    rtncipher.scale() = std::pow(2.0, 46);
}

}
