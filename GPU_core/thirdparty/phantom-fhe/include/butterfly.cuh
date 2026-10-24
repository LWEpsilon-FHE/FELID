#pragma once

#include "uintmodmath.cuh"

namespace phantom::arith {
     
    __device__ __forceinline__ void ct_butterfly(uint64_t& x, uint64_t& y,
                                                 const uint64_t& tw, const uint64_t& tw_shoup,
                                                 const uint64_t& mod) {
         
        const uint64_t hi = __umul64hi(y, tw_shoup);
        const uint64_t tw_y = y * tw - hi * mod;
         
        const uint64_t mod2 = 2 * mod;
        const uint64_t tmp = x - mod2;
        x = tmp + (tmp >> 63) * mod2;
        y = x + mod2 - tw_y;
        x += tw_y;
    }

    __device__ __forceinline__ void gs_butterfly(uint64_t& x, uint64_t& y,
                                                 const uint64_t& tw, const uint64_t& tw_shoup,
                                                 const uint64_t& mod) {
        const uint64_t mod2 = 2 * mod;
        const uint64_t t = x + mod2 - y;  
        uint64_t s = x + y;  
        csub_q(s, mod2);  
        x = s;
        y = multiply_and_reduce_shoup_lazy(t, tw, tw_shoup, mod);
    }

    __device__ __forceinline__ void fntt8(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
         
        ct_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
         
        ct_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        ct_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
         
        ct_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
        ct_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
        ct_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
        ct_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);
    }

    __device__ __forceinline__ void fntt4(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
         
        ct_butterfly(s[0], s[2], tw[tw_idx], tw_shoup[tw_idx], mod);
        ct_butterfly(s[1], s[3], tw[tw_idx], tw_shoup[tw_idx], mod);
         
        ct_butterfly(s[0], s[1], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        ct_butterfly(s[2], s[3], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
    }

    __device__ __forceinline__ void intt8(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
         
        gs_butterfly(s[0], s[1], tw[4 * tw_idx], tw_shoup[4 * tw_idx], mod);
        gs_butterfly(s[2], s[3], tw[4 * tw_idx + 1], tw_shoup[4 * tw_idx + 1], mod);
        gs_butterfly(s[4], s[5], tw[4 * tw_idx + 2], tw_shoup[4 * tw_idx + 2], mod);
        gs_butterfly(s[6], s[7], tw[4 * tw_idx + 3], tw_shoup[4 * tw_idx + 3], mod);

        gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[1], s[3], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
        gs_butterfly(s[5], s[7], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);

        gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[1], s[5], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[3], s[7], tw[tw_idx], tw_shoup[tw_idx], mod);
    }

    __device__ __forceinline__ void intt4(uint64_t* s,
                                          const uint64_t* tw,
                                          const uint64_t* tw_shoup,
                                          uint64_t tw_idx,
                                          uint64_t mod) {
         
        gs_butterfly(s[0], s[2], tw[2 * tw_idx], tw_shoup[2 * tw_idx], mod);
        gs_butterfly(s[4], s[6], tw[2 * tw_idx + 1], tw_shoup[2 * tw_idx + 1], mod);
         
        gs_butterfly(s[0], s[4], tw[tw_idx], tw_shoup[tw_idx], mod);
        gs_butterfly(s[2], s[6], tw[tw_idx], tw_shoup[tw_idx], mod);
    }
}
