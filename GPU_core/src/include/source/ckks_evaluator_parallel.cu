#include <iostream>
#include <cstdlib>

#include "ckks_evaluator_parallel.cuh"
#include "phantom.h"
#include "utils.cuh"

using namespace phantom::arith;
using namespace phantom::util;
using namespace moai;

namespace {

}

vector<double> CKKSEvaluator::init_vec_with_value(double value) {
  std::vector<double> vec(encoder.slot_count(), value);
  return vec;
}

PhantomPlaintext CKKSEvaluator::init_plain_power_of_x(size_t exponent) {
  PhantomPlaintext plain_power_of_x;

  vector<double> vec = init_vec_with_value(0.0);
  vec[exponent] = 1.0;

  encoder.encode(vec, scale, plain_power_of_x);
  return plain_power_of_x;
}

PhantomCiphertext CKKSEvaluator::init_guess(PhantomCiphertext x) {
  PhantomPlaintext A, B;
  encoder.encode(-1.29054537e-04, scale, A);
  encoder.encode(1.29054537e-01, scale, B);
  return eval_line(x, A, B);
}

PhantomCiphertext CKKSEvaluator::eval_line(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c) {
  evaluator.mod_switch_to_inplace(m, x.params_id());
  evaluator.multiply_plain_inplace(x, m);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(c, x.params_id());
  x.scale() = scale;
  evaluator.add_plain_inplace(x, c);

  return x;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt(PhantomCiphertext x, int d_newt, int d_gold) {
  PhantomCiphertext res = init_guess(x);
  PhantomCiphertext y = newton_iter(x, res, d_newt);
  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, y, d_gold);
  return sqrt_inv_sqrt.second;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_boot(
    PhantomCiphertext x,
    int d_newt,
    int d_gold,
    const std::function<void(PhantomCiphertext &, const PhantomCiphertext &)> &bootstrap) {
  if (!bootstrap) {
    throw std::invalid_argument("invert_sqrt_boot requires a bootstrap callback");
  }
  const auto x_depth = static_cast<int>(context->get_context_data(x.params_id()).chain_depth());
  if (x_depth < 4) {
    throw std::invalid_argument("invert_sqrt_boot requires input depth at least 4");
  }

  PhantomCiphertext res = init_guess(x);
  
  const int newton_start_depth = x_depth - 4;
  while (static_cast<int>(context->get_context_data(res.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(res);
  }

  PhantomCiphertext y = newton_iter(x, res, d_newt);
  
  while (context->get_context_data(y.params_id()).chain_depth() > 0) {
    evaluator.mod_switch_to_next_inplace(y);
  }

  PhantomCiphertext refreshed_y;
  bootstrap(refreshed_y, y);
  
  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, refreshed_y, d_gold);
  
  return sqrt_inv_sqrt.second;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_boot2(
    PhantomCiphertext x,
    int d_newt,
    int d_gold,
    const std::function<void(PhantomCiphertext &, const PhantomCiphertext &)> &bootstrap) {
  if (!bootstrap) {
    throw std::invalid_argument("invert_sqrt_boot2 requires a bootstrap callback");
  }
  const auto x_depth = static_cast<int>(context->get_context_data(x.params_id()).chain_depth());
  if (x_depth < 2) {
    throw std::invalid_argument("invert_sqrt_boot2 requires input depth at least 2");
  }

  PhantomCiphertext res = init_guess(x);
  
  const int newton_start_depth = x_depth - 2;
  while (static_cast<int>(context->get_context_data(res.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(res);
  }

  PhantomCiphertext y = newton_iter(x, res, d_newt);
  
  while (context->get_context_data(y.params_id()).chain_depth() > 0) {
    evaluator.mod_switch_to_next_inplace(y);
  }

  PhantomCiphertext refreshed_y;
  bootstrap(refreshed_y, y);
  
  while (static_cast<int>(context->get_context_data(refreshed_y.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(refreshed_y);
  }

  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, refreshed_y, d_gold);
  
  return sqrt_inv_sqrt.second;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_boot_unit_range(
    PhantomCiphertext x,
    int d_newt,
    int d_gold,
    const std::function<void(PhantomCiphertext &, const PhantomCiphertext &)> &bootstrap) {
  if (!bootstrap) {
    throw std::invalid_argument("invert_sqrt_boot_unit_range requires a bootstrap callback");
  }
  const auto x_depth = static_cast<int>(context->get_context_data(x.params_id()).chain_depth());
  if (x_depth < 4) {
    throw std::invalid_argument("invert_sqrt_boot_unit_range requires input depth at least 4");
  }

  PhantomPlaintext slope, intercept;
  encoder.encode(-2.0, scale, slope);
  encoder.encode(3.0, scale, intercept);
  PhantomCiphertext res = eval_line(x, slope, intercept);
  
  const int newton_start_depth = x_depth - 4;
  while (static_cast<int>(context->get_context_data(res.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(res);
  }

  PhantomCiphertext y = newton_iter(x, res, d_newt);
  
  while (context->get_context_data(y.params_id()).chain_depth() > 0) {
    evaluator.mod_switch_to_next_inplace(y);
  }

  PhantomCiphertext refreshed_y;
  bootstrap(refreshed_y, y);
  
  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, refreshed_y, d_gold);
  
  return sqrt_inv_sqrt.second;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_boot2_unit_range(
    PhantomCiphertext x,
    int d_newt,
    int d_gold,
    const std::function<void(PhantomCiphertext &, const PhantomCiphertext &)> &bootstrap) {
  if (!bootstrap) {
    throw std::invalid_argument("invert_sqrt_boot2_unit_range requires a bootstrap callback");
  }
  const auto x_depth = static_cast<int>(context->get_context_data(x.params_id()).chain_depth());
  if (x_depth < 2) {
    throw std::invalid_argument("invert_sqrt_boot2_unit_range requires input depth at least 2");
  }

  PhantomPlaintext slope, intercept;
  encoder.encode(-2.0, scale, slope);
  encoder.encode(3.0, scale, intercept);
  PhantomCiphertext res = eval_line(x, slope, intercept);
  
  const int newton_start_depth = x_depth - 2;
  while (static_cast<int>(context->get_context_data(res.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(res);
  }

  PhantomCiphertext y = newton_iter(x, res, d_newt);
  
  while (context->get_context_data(y.params_id()).chain_depth() > 0) {
    evaluator.mod_switch_to_next_inplace(y);
  }

  PhantomCiphertext refreshed_y;
  bootstrap(refreshed_y, y);
  
  while (static_cast<int>(context->get_context_data(refreshed_y.params_id()).chain_depth()) > newton_start_depth) {
    evaluator.mod_switch_to_next_inplace(refreshed_y);
  }

  pair<PhantomCiphertext, PhantomCiphertext> sqrt_inv_sqrt = goldschmidt_iter(x, refreshed_y, d_gold);
  
  return sqrt_inv_sqrt.second;
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_unit_newton(PhantomCiphertext x, int d_newt) {
  PhantomPlaintext zero, one;
  encoder.encode(0.0, x.chain_index(), scale, zero);
  encoder.encode(1.0, x.chain_index(), scale, one);
  PhantomCiphertext res = eval_line(x, zero, one);
  return newton_iter(x, res, d_newt);
}

PhantomCiphertext CKKSEvaluator::invert_sqrt_unit_newton_with_initial(PhantomCiphertext x, PhantomCiphertext initial, int d_newt) {
  return newton_iter(x, initial, d_newt);
}

uint64_t CKKSEvaluator::get_modulus(PhantomCiphertext &x, int k) {
  const vector<phantom::arith::Modulus> &modulus = context->get_context_data(x.params_id()).parms().coeff_modulus();
  int sz = modulus.size();
  return modulus[sz - k].value();
}

void CKKSEvaluator::re_encrypt(PhantomCiphertext &ct) {
  
  while (ct.coeff_modulus_size() > 1) {
    evaluator.mod_switch_to_next_inplace(ct);
  }

  PhantomPlaintext temp;
  vector<double> v;
  decryptor.decrypt(ct, temp);
  encoder.decode(temp, v);
  encoder.encode(v, ct.scale(), temp);
  encryptor.encrypt(temp, ct);

}

pair<PhantomCiphertext, PhantomCiphertext> CKKSEvaluator::goldschmidt_iter(PhantomCiphertext v, PhantomCiphertext y, int d) {
  PhantomCiphertext x, h, r, temp;
  PhantomPlaintext constant;

  encoder.encode(0.5, scale, constant);

  evaluator.mod_switch_to_inplace(v, y.params_id());
  evaluator.multiply(v, y, x);
  evaluator.relinearize_inplace(x, *relin_keys);
  evaluator.rescale_to_next_inplace(x);

  evaluator.mod_switch_to_inplace(constant, y.params_id());
  evaluator.multiply_plain(y, constant, h);
  evaluator.rescale_to_next_inplace(h);

  for (int i = 0; i < d; i++) {
    encoder.encode(0.5, scale, constant);

    evaluator.multiply(x, h, r);
    evaluator.relinearize_inplace(r, *relin_keys);
    evaluator.rescale_to_next_inplace(r);

    r.scale() = scale;
    evaluator.negate(r, temp);
    evaluator.mod_switch_to_inplace(constant, temp.params_id());
    evaluator.add_plain(temp, constant, r);

    evaluator.mod_switch_to_inplace(x, r.params_id());
    evaluator.multiply(x, r, temp);
    evaluator.relinearize_inplace(temp, *relin_keys);
    evaluator.rescale_to_next_inplace(temp);

    x.scale() = scale;
    temp.scale() = scale;
    evaluator.mod_switch_to_inplace(x, temp.params_id());
    evaluator.add_inplace(x, temp);

    evaluator.mod_switch_to_inplace(h, r.params_id());
    evaluator.multiply(h, r, temp);
    evaluator.relinearize_inplace(temp, *relin_keys);
    evaluator.rescale_to_next_inplace(temp);

    h.scale() = scale;
    temp.scale() = scale;
    evaluator.mod_switch_to_inplace(h, temp.params_id());
    evaluator.add_inplace(h, temp);
  }

  encoder.encode(2.0, scale, constant);
  evaluator.mod_switch_to_inplace(constant, h.params_id());
  evaluator.multiply_plain_inplace(h, constant);
  evaluator.rescale_to_next_inplace(h);

  return make_pair(x, h);
}

PhantomCiphertext CKKSEvaluator::newton_iter(PhantomCiphertext x, PhantomCiphertext res, int iter) {
  for (int i = 0; i < iter; i++) {
    PhantomPlaintext three_half, neg_half;

    encoder.encode(1.5, scale, three_half);
    encoder.encode(-0.5, scale, neg_half);

    PhantomCiphertext res_sq;
    evaluator.square(res, res_sq);
    evaluator.relinearize_inplace(res_sq, *relin_keys);
    evaluator.rescale_to_next_inplace(res_sq);

    PhantomCiphertext res_x;
    evaluator.mod_switch_to_inplace(neg_half, x.params_id());
    evaluator.multiply_plain(x, neg_half, res_x);
    evaluator.rescale_to_next_inplace(res_x);

    if (context->get_context_data(res.params_id()).chain_depth() <
        context->get_context_data(res_x.params_id()).chain_depth())
      evaluator.mod_switch_to_inplace(res_x, res.params_id());
    else
      evaluator.mod_switch_to_inplace(res, res_x.params_id());

    evaluator.multiply_inplace(res_x, res);
    evaluator.relinearize_inplace(res_x, *relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(res_sq, res_x.params_id());
    evaluator.multiply_inplace(res_x, res_sq);
    evaluator.relinearize_inplace(res_x, *relin_keys);
    evaluator.rescale_to_next_inplace(res_x);

    evaluator.mod_switch_to_inplace(three_half, res.params_id());
    evaluator.multiply_plain_inplace(res, three_half);
    evaluator.rescale_to_next_inplace(res);

    evaluator.mod_switch_to_inplace(res, res_x.params_id());
    res_x.scale() = scale;
    res.scale() = scale;
    evaluator.add_inplace(res, res_x);
  }

  return res;
}

void CKKSEvaluator::eval_odd_deg9_poly(vector<double> &a, PhantomCiphertext &x, PhantomCiphertext &dest) {

  double D = x.scale();   

  uint64_t p = get_modulus(x, 1);
  uint64_t q = get_modulus(x, 2);
  uint64_t r = get_modulus(x, 3);
  uint64_t s = get_modulus(x, 4);
  uint64_t t = get_modulus(x, 5);

  p = q;
  q = r;
  r = s;
  s = t;

  vector<double> a_scales(10);
  a_scales[1] = q;
  a_scales[3] = (double)p / D * q / D * r;
  a_scales[5] = (double)p / D * p / D * q / D * q / D * r;
  a_scales[7] = (double)p / D * p / D * q / D * q / D * q / D * r / D * s;
  a_scales[9] = (double)p / D * p / D * p / D * q / D * q / D * q / D * r / D * r / D * s;

  PhantomCiphertext x2, x3, x6;

  evaluator.square(x, x2);
  evaluator.relinearize_inplace(x2, *relin_keys);
  evaluator.rescale_to_next_inplace(x2);   

  evaluator.mod_switch_to_next_inplace(x);   
  evaluator.multiply(x2, x, x3);
  evaluator.relinearize_inplace(x3, *relin_keys);
  evaluator.rescale_to_next_inplace(x3);   

  evaluator.square(x3, x6);
  evaluator.relinearize_inplace(x6, *relin_keys);
  evaluator.rescale_to_next_inplace(x6);   

  PhantomPlaintext a1, a3, a5, a7, a9;

  PhantomCiphertext T1;
  double a5_scale = D / x2.scale() * p / x3.scale() * q;
  encoder.encode(a[5], x2.params_id(), a5_scale, a5);   
  evaluator.multiply_plain(x2, a5, T1);
  evaluator.rescale_to_next_inplace(T1);   

  encoder.encode(a[3], T1.params_id(), T1.scale(), a3);   

  evaluator.add_plain_inplace(T1, a3);   
  evaluator.multiply_inplace(T1, x3);
  evaluator.relinearize_inplace(T1, *relin_keys);
  evaluator.rescale_to_next_inplace(T1);   

  PhantomCiphertext T2;
  PhantomPlaintext a9_switched;
  double a9_scale = D / x3.scale() * r / x6.scale() * q;
  encoder.encode(a[9], x3.params_id(), a9_scale, a9);   
  evaluator.multiply_plain(x3, a9, T2);
  evaluator.rescale_to_next_inplace(T2);   

  PhantomCiphertext a7x;
  double a7_scale = T2.scale() / x.scale() * p;
  encoder.encode(a[7], x.params_id(), a7_scale, a7);   
  evaluator.multiply_plain(x, a7, a7x);
  evaluator.rescale_to_next_inplace(a7x);                 
  evaluator.mod_switch_to_inplace(a7x, T2.params_id());   

  double mid_scale = (T2.scale() + a7x.scale()) / 2;
  T2.scale() = a7x.scale() = mid_scale;   
  evaluator.add_inplace(T2, a7x);         
  evaluator.multiply_inplace(T2, x6);
  evaluator.relinearize_inplace(T2, *relin_keys);
  evaluator.rescale_to_next_inplace(T2);   

  PhantomCiphertext T3;
  encoder.encode(a[1], x.params_id(), p, a1);   
  evaluator.multiply_plain(x, a1, T3);
  evaluator.rescale_to_next_inplace(T3);   

  double mid3_scale = (T1.scale() + T2.scale() + T3.scale()) / 3;
  T1.scale() = T2.scale() = T3.scale() = mid3_scale;

  dest = T2;
  evaluator.mod_switch_to_inplace(T1, dest.params_id());   
  evaluator.add_inplace(dest, T1);
  evaluator.mod_switch_to_inplace(T3, dest.params_id());   
  evaluator.add_inplace(dest, T3);

}

PhantomCiphertext CKKSEvaluator::sgn_eval(PhantomCiphertext x, int d_g, int d_f, double sgn_factor) {
   
  vector<double> f4_coeffs = F4_COEFFS;
  vector<double> g4_coeffs = G4_COEFFS;
  vector<double> f4_coeffs_last(10, 0.0);
  vector<double> g4_coeffs_last(10, 0.0);

  for (int i = 0; i <= 9; i++) {
    f4_coeffs[i] /= F4_SCALE;
    f4_coeffs_last[i] = f4_coeffs[i] * sgn_factor;

    g4_coeffs[i] /= G4_SCALE;
    g4_coeffs_last[i] = g4_coeffs[i] * sgn_factor;
  }

  PhantomCiphertext dest = x;

  for (int i = 0; i < d_g; i++) {
    if (i == d_g - 1) {
      eval_odd_deg9_poly(g4_coeffs_last, dest, dest);
    } else {
      eval_odd_deg9_poly(g4_coeffs, dest, dest);
    }
  }

  for (int i = 0; i < d_f; i++) {
    if (i == d_f - 1) {
      eval_odd_deg9_poly(f4_coeffs_last, dest, dest);
    } else {
      eval_odd_deg9_poly(f4_coeffs, dest, dest);
    }
  }

  return dest;
}

double CKKSEvaluator::calculate_MAE(vector<double> &y_true, PhantomCiphertext &ct, int N) {
  PhantomPlaintext temp;
  vector<double> y_pred;

  decryptor.decrypt(ct, temp);
  encoder.decode(temp, y_pred);

  double sum_absolute_errors = 0.0;
  for (size_t i = 0; i < N; ++i) {
    sum_absolute_errors += abs(y_true[i] - y_pred[i]);
  }

  return sum_absolute_errors / N;
}

PhantomCiphertext CKKSEvaluator::exp(PhantomCiphertext x) {
  PhantomPlaintext one, inverse_128;

  encoder.encode(0.0078125, x.params_id(), x.scale(), inverse_128);
  evaluator.multiply_plain_inplace(x, inverse_128);
  evaluator.rescale_to_next_inplace(x);

  encoder.encode(1.0, x.params_id(), x.scale(), one);
  evaluator.add_plain_inplace(x, one);

  for (int i = 0; i < log2(128); i++) {
    evaluator.square(x, x);
    evaluator.relinearize_inplace(x, *relin_keys);
    evaluator.rescale_to_next_inplace(x);
  }

  return x;
}

PhantomCiphertext CKKSEvaluator::inverse(PhantomCiphertext x, int iter) {
  PhantomCiphertext y, tmp, res;
  PhantomPlaintext one;

  encoder.encode(1.0, x.params_id(), x.scale(), one);
  evaluator.sub_plain(x, one, y);
  evaluator.negate_inplace(y);
  evaluator.add_plain(y, one, tmp);

  res = tmp;

  for (int i = 0; i < iter; i++) {
    evaluator.square_inplace(y);
    evaluator.relinearize_inplace(y, *relin_keys);
    evaluator.rescale_to_next_inplace(y);

    encoder.encode(1.0, y.params_id(), y.scale(), one);
    evaluator.add_plain(y, one, tmp);

    evaluator.mod_switch_to_inplace(res, tmp.params_id());
    evaluator.multiply_inplace(res, tmp);
    evaluator.relinearize_inplace(res, *relin_keys);
    evaluator.rescale_to_next_inplace(res);
  }

  return res;
}

void moai::Evaluator::add_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const cuda_stream_wrapper &stream_wrapper) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.scale() = encrypted2.scale();
    add_inplace(encrypted1, encrypted2, stream_wrapper);
    return;
  }

  else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    auto &context_data = context->get_context_data(encrypted2.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted, stream_wrapper);
    encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted2_adjusted, stream_wrapper);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.params_id(), stream_wrapper);
    encrypted1.scale() = encrypted2_adjusted.scale();
    add_inplace(encrypted1, encrypted2_adjusted, stream_wrapper);
  }

  else {
    auto &context_data = context->get_context_data(encrypted1.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted, stream_wrapper);
    encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted1_adjusted, stream_wrapper);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.params_id(), stream_wrapper);
    encrypted1_adjusted.scale() = encrypted2.scale();

    is_same_stream(encrypted1_adjusted.data_ptr().get_stream(), stream_wrapper.get_stream());
    add(encrypted1_adjusted, encrypted2, encrypted1, stream_wrapper);
  }
}

void moai::Evaluator::sub_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const cuda_stream_wrapper &stream_wrapper) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.scale() = encrypted2.scale();
    sub_inplace(encrypted1, encrypted2);
    return;
  }

  else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    auto &context_data = context->get_context_data(encrypted2.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
    encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted2_adjusted);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.params_id());
    encrypted1.scale() = encrypted2_adjusted.scale();
    sub_inplace(encrypted1, encrypted2_adjusted);
  }

  else {
    auto &context_data = context->get_context_data(encrypted1.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
    encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted1_adjusted);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.params_id());
    encrypted1_adjusted.scale() = encrypted2.scale();
    sub(encrypted1_adjusted, encrypted2, encrypted1);
  }
}

void moai::Evaluator::multiply_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys, const cuda_stream_wrapper &stream_wrapper) {
  size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
  size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();

  if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
    encrypted1.scale() = encrypted2.scale();
    multiply_inplace(encrypted1, encrypted2);
    relinearize_inplace(encrypted1, relin_keys);
    return;
  }

  else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
    auto &context_data = context->get_context_data(encrypted2.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted2_adjusted;

    double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
    multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
    encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted2_adjusted);
    mod_switch_to_inplace(encrypted2_adjusted, encrypted1.params_id());
    encrypted1.scale() = encrypted2_adjusted.scale();
    multiply_inplace(encrypted1, encrypted2_adjusted);
  }

  else {
    auto &context_data = context->get_context_data(encrypted1.params_id());
    auto &parms = context_data.parms();
    auto modulus = parms.coeff_modulus();
    PhantomCiphertext encrypted1_adjusted;

    double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
    multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
    encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
    rescale_to_next_inplace(encrypted1_adjusted);
    mod_switch_to_inplace(encrypted1_adjusted, encrypted2.params_id());
    encrypted1_adjusted.scale() = encrypted2.scale();
    multiply(encrypted1_adjusted, encrypted2, encrypted1);
  }

  relinearize_inplace(encrypted1, relin_keys);
}
