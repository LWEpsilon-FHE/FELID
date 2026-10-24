#pragma once

#include "common/Remez.cuh"

class RemezGeluAlgo6Even : public boot::Remez {
 public:
  RemezGeluAlgo6Even(RemezParam _param, double _log_width, long _deg)
      : boot::Remez(_param, 1, _log_width, _deg) {}

  RR function_value(RR x) override {
    const RR alpha = sqrt(RR(2) / ComputePi_RR());
    const RR beta(NTL::INIT_VAL_TYPE{}, "0.044715");
    const RR pi = ComputePi_RR();
    const RR quarter = RR(0.25);
    const RR half = RR(0.5);
    const RR x_abs = (arcsin(x) / (RR(2) * pi) + quarter) * half;
    const RR cubic = x_abs * x_abs * x_abs;
    const RR inner = alpha * (x_abs + beta * cubic);
    const RR exp_pos = exp(inner);
    const RR exp_neg = exp(-inner);
    const RR tanh_inner = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    return half * x_abs * tanh_inner;
  }
};
