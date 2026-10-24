#pragma once

#include "uintmodmath.cuh"
#include "gputype.h"

__global__ void negate_rns_poly(const uint64_t* operand,
                                const DModulus* modulus,
                                uint64_t* result,
                                uint64_t poly_degree,
                                uint64_t coeff_mod_size);

__global__ void add_rns_poly(const uint64_t* operand1,
                             const uint64_t* operand2,
                             const DModulus* modulus,
                             uint64_t* result,
                             uint64_t poly_degree,
                             uint64_t coeff_mod_size);

__global__ void add_std_cipher(const uint64_t* cipher1,
                               const uint64_t* cipher2,
                               const DModulus* modulus,
                               uint64_t* result,
                               uint64_t poly_degree,
                               uint64_t coeff_mod_size);

__global__ void sub_rns_poly(const uint64_t* operand1,
                             const uint64_t* operand2,
                             const DModulus* modulus,
                             uint64_t* result,
                             uint64_t poly_degree,
                             uint64_t coeff_mod_size);

__global__ void add_and_negate_rns_poly(const uint64_t* operand1,
                                        const uint64_t* operand2,
                                        const DModulus* modulus,
                                        uint64_t* result,
                                        uint64_t poly_degree,
                                        uint64_t coeff_mod_size);

__global__ void add_many_rns_poly(const uint64_t* const * operands,
                                  uint64_t add_size,
                                  const DModulus* modulus,
                                  uint64_t* result,
                                  uint32_t poly_index,
                                  uint32_t poly_degree,
                                  uint32_t coeff_mod_size,
                                  uint64_t reduction_threshold);

__global__ void multiply_rns_poly(const uint64_t* operand1,
                                  const uint64_t* operand2,
                                  const DModulus* modulus,
                                  uint64_t* result,
                                  uint64_t poly_degree,
                                  uint64_t coeff_mod_size);

__global__ void multiply_scalar_rns_poly(const uint64_t* operand,
                                         uint64_t scalar,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint64_t poly_degree,
                                         uint64_t coeff_mod_size);

__global__ void multiply_scalar_rns_poly(const uint64_t* operand,
                                         const uint64_t* scalar,
                                         const uint64_t* scalar_shoup,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint64_t poly_degree,
                                         uint64_t coeff_mod_size);

__global__ void multiply_and_add_rns_poly(const uint64_t* operand1,
                                          const uint64_t* operand2,
                                          const uint64_t* operand3,
                                          const DModulus* modulus,
                                          uint64_t* result,
                                          uint64_t poly_degree,
                                          uint64_t coeff_mod_size);

__global__ void multiply_scalar_and_add_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t scalar,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size);

__global__ void multiply_scalar_and_sub_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t scalar,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 const uint64_t poly_degree,
                                                 const uint64_t coeff_mod_size);

__global__ void multiply_and_scale_add_rns_poly(const uint64_t* operand1,
                                                const uint64_t* operand2,
                                                const uint64_t* operand3,
                                                const uint64_t& scale,
                                                const DModulus* modulus,
                                                uint64_t* result,
                                                const uint64_t poly_degree,
                                                const uint64_t coeff_mod_size);

__global__ void multiply_temp_mod_and_add_rns_poly(const uint64_t* operand1,
                                                   const uint64_t* const * operand2,
                                                   const DModulus* modulus,
                                                   uint64_t** result,
                                                   size_t n,
                                                   size_t dnum,
                                                   size_t alpha,
                                                   const uint64_t* factor,
                                                   const uint64_t* factor_shoup);

__global__ void multiply_and_add_negate_rns_poly(const uint64_t* operand1,
                                                 const uint64_t* operand2,
                                                 const uint64_t* operand3,
                                                 const DModulus* modulus,
                                                 uint64_t* result,
                                                 uint64_t poly_degree,
                                                 uint64_t coeff_mod_size);

__global__ void sub_and_scale_single_mod_poly(const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t scale,
                                              uint64_t scale_shoup,
                                              uint64_t modulus,
                                              uint64_t* result,
                                              uint64_t poly_degree);

__global__ void sub_and_scale_rns_poly(const uint64_t* operand1,
                                       const uint64_t* operand2,
                                       const uint64_t* scale,
                                       const uint64_t* scale_shoup,
                                       const DModulus* modulus,
                                       uint64_t* result,
                                       uint64_t poly_degree,
                                       uint64_t coeff_mod_size);

__global__ void bfv_add_timesQ_overt_kernel(uint64_t* ct,
                                            const uint64_t* pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t* tInv_mod_q,
                                            const uint64_t* tInv_mod_q_shoup,
                                            const DModulus* modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql);

__global__ void bfv_sub_timesQ_overt_kernel(uint64_t* ct,
                                            const uint64_t* pt,
                                            uint64_t negQl_mod_t,
                                            uint64_t negQl_mod_t_shoup,
                                            const uint64_t* tInv_mod_q,
                                            const uint64_t* tInv_mod_q_shoup,
                                            const DModulus* modulus_Ql,
                                            uint64_t t,
                                            uint64_t n,
                                            uint64_t size_Ql);

__global__ void multiply_and_negated_add_rns_poly(const uint64_t* operand1, uint64_t negate,
                                                  const uint64_t* operand2,
                                                  const uint64_t* operand3,
                                                  const DModulus* modulus,
                                                  uint64_t* result,
                                                  uint64_t poly_degree,
                                                  uint64_t coeff_mod_size);

__global__ void abs_plain_rns_poly(const uint64_t* operand,
                                   uint64_t plain_upper_half_threshold,
                                   const uint64_t* plain_upper_half_increment,
                                   uint64_t* result,
                                   uint64_t poly_degree,
                                   uint64_t coeff_mod_size);

__global__ void tensor_prod_2x2_rns_poly(const uint64_t* operand1,
                                         const uint64_t* operand2,
                                         const DModulus* modulus,
                                         uint64_t* result,
                                         uint32_t poly_degree,
                                         uint32_t coeff_mod_size);

__global__ void tensor_prod_2x2_rns_poly_lazy(const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t* result,
                                              uint32_t poly_degree,
                                              uint32_t coeff_mod_size);

__global__ void tensor_square_2x2_rns_poly(const uint64_t* operand,
                                           const DModulus* modulus,
                                           uint64_t* result,
                                           const uint32_t poly_degree,
                                           const uint32_t coeff_mod_size);

__global__ void tensor_prod_mxn_rns_poly(const uint64_t* operand1, uint32_t op1_size,
                                         const uint64_t* operand2, uint32_t op2_size,
                                         const DModulus* modulus,
                                         uint64_t* result, uint64_t res_size,
                                         uint32_t poly_degree, uint32_t coeff_mod_size);
