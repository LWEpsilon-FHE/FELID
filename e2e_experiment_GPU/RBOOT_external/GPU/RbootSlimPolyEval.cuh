#pragma once

#include <NTL/RR.h>

#include <memory>
#include <vector>

#include "source/bootstrapping/common/PolyUpdate.cuh"
#include "source/bootstrapping/common/Polynomial.cuh"
#include "RbootSparseOps.cuh"

namespace rboot_slim {

long pow2(long value);
long ceil_to_int(double value);

void upgrade_oddbaby(long n, minicomp::Tree &tree);
void upgrade_baby(long n, minicomp::Tree &tree);
long coeff_number(long deg, minicomp::Tree &tree);

void homomorphic_poly_evaluation_sparse(const boot::Polynomial &poly,
                                        const RbootSparseOps &ops,
                                        PhantomCiphertext &rtn,
                                        const PhantomCiphertext &cipher,
                                        const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

void eval_polynomial_integrate(const RbootSparseOps &ops,
                               PhantomCiphertext &res,
                               const PhantomCiphertext &cipher,
                               long deg,
                               const std::vector<NTL::RR> &decomp_coeff,
                               minicomp::Tree &tree,
                               const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream);

}
