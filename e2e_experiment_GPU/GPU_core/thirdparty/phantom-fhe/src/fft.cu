#include "fft.h"
#include "context.cuh"

using namespace phantom::arith;

namespace phantom::util {

    constexpr double ComplexRoots::PI_;

    ComplexRoots::ComplexRoots(size_t degree_of_roots) : degree_of_roots_(degree_of_roots) {
        roots_ = (cuDoubleComplex *) malloc((degree_of_roots_ / 8 + 1) * sizeof(cuDoubleComplex));

        for (size_t i = 0; i <= degree_of_roots_ / 8; i++) {
            roots_[i] = polar(1.0, 2 * PI_ * static_cast<double>(i) / static_cast<double>(degree_of_roots_));
        }
    }

    cuDoubleComplex ComplexRoots::get_root(size_t index) const {
        index &= degree_of_roots_ - 1;
        auto mirror = [](cuDoubleComplex a) {
            return make_cuDoubleComplex(a.y, a.x);
        };

        if (index <= degree_of_roots_ / 8) {
            return roots_[index];
        } else if (index <= degree_of_roots_ / 4) {
            return mirror(roots_[degree_of_roots_ / 4 - index]);
        } else if (index <= degree_of_roots_ / 2) {
            return cuCsub({0, 0}, cuConj(get_root(degree_of_roots_ / 2 - index)));
        } else if (index <= 3 * degree_of_roots_ / 4) {
            return cuCsub({0, 0}, get_root(index - degree_of_roots_ / 2));
        } else {
            return cuConj(get_root(degree_of_roots_ - index));
        }
    }
}

__device__ __forceinline__ void ct_butterfly_cplx(cuDoubleComplex *x,
                                                  const cuDoubleComplex &pow) {
    cuDoubleComplex s[2];
    s[0] = x[0];
    s[1] = cuCmul(x[1], pow);
    x[0] = cuCadd(s[0], s[1]);
    x[1] = cuCsub(s[0], s[1]);
}

__device__ __forceinline__ void gs_butterfly_cplx(cuDoubleComplex *x,
                                                  const cuDoubleComplex &pow) {
    cuDoubleComplex s[2];

    s[0] = cuCadd(x[0], x[1]);
    s[1] = cuCsub(x[0], x[1]);

    x[0] = s[0];
    x[1] = cuCmul(s[1], pow);
}

__global__ void inplace_special_ffft_base_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const uint32_t iter,
                                                 const uint32_t M, const uint32_t logRatio) {
    extern __shared__ cuDoubleComplex buffer[];

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);  
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx, bufIdx;  
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];
         
        uint32_t psiIdx;
         
        int _iter = iter;
        int _numOfGroups = numOfGroups;
        for (; _numOfGroups < n; _numOfGroups <<= 1) {
            logPairs = logn - _iter - 1;
            pairsInGroup = 1 << logPairs;

            k = tid >> logPairs;
            j = tid & (pairsInGroup - 1);
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx - blockIdx.x * (n >> iter);

            psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
            psiIdx <<= logPairs + logRatio;
            psiIdx &= M - 1;  
             
            one_twiddle = twiddles[psiIdx];

            if (_numOfGroups == numOfGroups) {
                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            ct_butterfly_cplx(samples, one_twiddle);

            _iter += 1;

            if (_numOfGroups == n >> 1) {

                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_special_ffft_iter_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const uint32_t iter,
                                                 const uint32_t M, const uint32_t logRatio) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];

        logPairs = logn - iter - 1;
        pairsInGroup = 1 << logPairs;

        k = tid >> logPairs;
        j = tid & (pairsInGroup - 1);
        glbIdx = 2 * k * pairsInGroup + j;

        uint32_t psiIdx;
        psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
        psiIdx <<= logPairs + logRatio;
        psiIdx &= M - 1;  
        one_twiddle = twiddles[psiIdx];

        samples[0] = inout[glbIdx];
        samples[1] = inout[glbIdx + pairsInGroup];

        ct_butterfly_cplx(samples, one_twiddle);

        inout[glbIdx] = samples[0];
        inout[glbIdx + pairsInGroup] = samples[1];
    }
}

__global__ void inplace_special_ifft_base_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const int32_t iter,
                                                 const uint32_t M, const uint32_t logRatio,
                                                 double scalar) {
    extern __shared__ cuDoubleComplex buffer[];

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx, bufIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];
        int _iter = logn - 1;
        int _numOfGroups = n >> 1;
         
        uint32_t psiIdx;
         
        for (; _numOfGroups >= numOfGroups; _numOfGroups >>= 1) {
            logPairs = logn - _iter - 1;
            pairsInGroup = 1 << logPairs;
            k = tid >> logPairs;
            j = tid & (pairsInGroup - 1);
            glbIdx = 2 * k * pairsInGroup + j;
            bufIdx = glbIdx - blockIdx.x * (n >> iter);

            psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
            psiIdx <<= logPairs + logRatio;
            psiIdx &= M - 1;  
             
            one_twiddle = twiddles[M - psiIdx];

            if (_numOfGroups == n >> 1) {

                samples[0] = inout[glbIdx];
                samples[1] = inout[glbIdx + pairsInGroup];
            } else {
                samples[0] = buffer[bufIdx];
                samples[1] = buffer[bufIdx + pairsInGroup];
            }

            gs_butterfly_cplx(samples, one_twiddle);
            _iter -= 1;
             
            if (_numOfGroups == 1) {
                if (scalar != 0.0) {
                    samples[0] = scalar_multiply_cuDoubleComplex(samples[0], scalar);
                    samples[1] = scalar_multiply_cuDoubleComplex(samples[1], scalar);
                }
            }

            if (_numOfGroups == numOfGroups) {
                inout[glbIdx] = samples[0];
                inout[glbIdx + pairsInGroup] = samples[1];
            } else {
                buffer[bufIdx] = samples[0];
                buffer[bufIdx + pairsInGroup] = samples[1];
                __syncthreads();
            }
        }
    }
}

__global__ void inplace_special_ifft_iter_kernel(cuDoubleComplex *inout,
                                                 const cuDoubleComplex *twiddles,
                                                 const uint32_t *group,
                                                 const uint32_t n, const uint32_t logn,
                                                 const uint32_t numOfGroups, const int32_t iter,
                                                 const uint32_t M, const uint32_t logRatio,
                                                 double scalar) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < (n >> 1);
         tid += blockDim.x * gridDim.x) {
        uint32_t logPairs, pairsInGroup;
        uint32_t k, j, glbIdx;
        cuDoubleComplex one_twiddle;
        cuDoubleComplex samples[2];

        logPairs = logn - iter - 1;
        pairsInGroup = 1 << logPairs;

        k = tid >> logPairs;
        j = tid & (pairsInGroup - 1);
        glbIdx = 2 * k * pairsInGroup + j;

        uint32_t psiIdx;
        psiIdx = group[__brev(k << logPairs) >> (33 - logn)];
        psiIdx <<= logPairs + logRatio;
        psiIdx &= M - 1;  
        one_twiddle = twiddles[M - psiIdx];

        samples[0] = inout[glbIdx];
        samples[1] = inout[glbIdx + pairsInGroup];

        gs_butterfly_cplx(samples, one_twiddle);
        if (numOfGroups == 1) {
            if (scalar != 0.0) {
                samples[0] = scalar_multiply_cuDoubleComplex(samples[0], scalar);
                samples[1] = scalar_multiply_cuDoubleComplex(samples[1], scalar);
            }
        }

        inout[glbIdx] = samples[0];
        inout[glbIdx + pairsInGroup] = samples[1];
    }
}

void special_fft_forward(DCKKSEncoderInfo &gp, const cudaStream_t &stream) {
    uint32_t threadsPerBlock, blocksPerGrid;

    if (gp.sparse_slots() == 00) {
        throw std::invalid_argument("the poly degree has not been set");
    }
    uint32_t logn = log2(gp.sparse_slots());
    uint32_t logRatio = log2(gp.m()) - logn - 2;

    if (gp.sparse_slots() <= SWITCH_POINT) {
         
        threadsPerBlock = gp.sparse_slots() >> 1;
        blocksPerGrid = 1;
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        inplace_special_ffft_base_kernel<<<blocksPerGrid, threadsPerBlock,
        gp.sparse_slots() * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                logRatio);
    } else {
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        threadsPerBlock = NTT_THREAD_PER_BLOCK;
        blocksPerGrid = ceil((float) gp.sparse_slots() / threadsPerBlock / 2);
        for (; numOfGroups < (gp.sparse_slots() / SWITCH_POINT); numOfGroups <<= 1) {
            inplace_special_ffft_iter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                    logRatio);

            iter++;
        }

        inplace_special_ffft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, SWITCH_POINT * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter,
                gp.m(), logRatio);
    }
}

void special_fft_backward(DCKKSEncoderInfo &gp, double scalar, const cudaStream_t &stream) {
    uint32_t threadsPerBlock, blocksPerGrid;
    uint32_t logn = log2(gp.sparse_slots());
    uint32_t logRatio = log2(gp.m()) - logn - 2;
    if (gp.sparse_slots() <= SWITCH_POINT) {
         
        uint32_t iter = 0;
        uint32_t numOfGroups = 1;
        threadsPerBlock = gp.sparse_slots() >> 1;
        blocksPerGrid = 1;
        inplace_special_ifft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, gp.sparse_slots() * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                logRatio, scalar);
    } else {
        int32_t iter = logn - log2(SWITCH_POINT);
        uint32_t numOfGroups = gp.sparse_slots() / SWITCH_POINT;
        if (iter < 0)
            iter = 0;
        if (numOfGroups < 1)
            numOfGroups = 1;

        threadsPerBlock = NTT_THREAD_PER_BLOCK;
        blocksPerGrid = ceil((float) gp.sparse_slots() / threadsPerBlock / 2);

        inplace_special_ifft_base_kernel<<<
        blocksPerGrid, threadsPerBlock, SWITCH_POINT * sizeof(cuDoubleComplex), stream>>>(
                gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter,
                gp.m(), logRatio, scalar);
        numOfGroups >>= 1;
        for (; numOfGroups >= 1; numOfGroups >>= 1) {
            iter--;

            inplace_special_ifft_iter_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    gp.in(), gp.twiddle(), gp.mul_group(), gp.sparse_slots(), logn, numOfGroups, iter, gp.m(),
                    logRatio, scalar);
        }
    }
}
