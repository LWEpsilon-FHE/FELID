#include "ckks.h"
#include "fft.h"
#include <cstring>
#include <list>
#include <unordered_map>

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;


namespace {
struct EncodingEntry {
    const PhantomContext *context;
    const PhantomCKKSEncoder *encoder;
    std::vector<cuDoubleComplex> values;
    size_t chain;
    uint32_t sparse_slots;
    double scale;
    PhantomPlaintext plain;
    uint64_t hash;
};
using EncodingEntries=std::list<EncodingEntry>;
thread_local EncodingEntries encoding_entries;
thread_local std::unordered_multimap<uint64_t,EncodingEntries::iterator> encoding_index;
thread_local size_t encoding_bytes=0;
thread_local unsigned long long cache_hits=0,cache_misses=0;
constexpr size_t encoding_limit=size_t(4)<<30;
uint64_t encoding_hash(const cuDoubleComplex *values,size_t count,size_t chain,double scale,uint32_t sparse_slots) {
    uint64_t hash=14695981039346656037ULL;
    auto append=[&](const void *pointer,size_t bytes) {
        const auto *data=static_cast<const unsigned char *>(pointer);
        for(size_t i=0;i<bytes;++i){hash^=data[i];hash*=1099511628211ULL;}
    };
    append(values,count*sizeof(*values));append(&chain,sizeof(chain));
    append(&scale,sizeof(scale));append(&sparse_slots,sizeof(sparse_slots));return hash;
}
void evict_encoding() {
    auto entry=std::prev(encoding_entries.end());
    auto range=encoding_index.equal_range(entry->hash);
    for(auto it=range.first;it!=range.second;++it)if(it->second==entry){encoding_index.erase(it);break;}
    encoding_bytes-=entry->plain.coeff_count()*sizeof(uint64_t);
    encoding_entries.erase(entry);
}
}
namespace phantom::execution {
void clear_encoding_cache() {
    Scope transfer(Phase::CacheTransfer);
    encoding_index.clear();encoding_entries.clear();encoding_bytes=0;
}
unsigned long long encoding_cache_hits(){return cache_hits;}
unsigned long long encoding_cache_misses(){return cache_misses;}
}

__global__ void bit_reverse_and_zero_padding(cuDoubleComplex *dst, cuDoubleComplex *src, uint64_t in_size,
                                             uint32_t slots, uint32_t logn) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x) {
        if (tid < uint32_t(in_size)) {
            dst[reverse_bits_uint32(tid, logn)] = src[tid];
        } else {
            dst[reverse_bits_uint32(tid, logn)] = (cuDoubleComplex) {0.0, 0.0};
        }
    }
}

__global__ void bit_reverse(cuDoubleComplex *dst, cuDoubleComplex *src, uint32_t slots, uint32_t logn) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x) {
        dst[reverse_bits_uint32(tid, logn)] = src[tid];
    }
}

PhantomCKKSEncoder::PhantomCKKSEncoder(const PhantomContext &context) {
    phantom::execution::Scope setup(phantom::execution::Phase::Setup);
    const auto &s = global_variables::default_stream->get_stream();

    auto &context_data = context.get_context_data(first_chain_index_);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (parms.scheme() != scheme_type::ckks) {
        throw std::invalid_argument("unsupported scheme");
    }
    slots_ = coeff_count >> 1;  

    auto specified_sparse_slots = context_data.parms().sparse_slots();
    if (specified_sparse_slots) {
         
        decoding_sparse_slots_ = specified_sparse_slots;
    }

    uint32_t m = coeff_count << 1;
    uint32_t slots_half = slots_ >> 1;
    gpu_ckks_msg_vec_ = std::make_unique<DCKKSEncoderInfo>(coeff_count, s);

    root_powers_.reserve(m);
    rotation_group_.reserve(slots_half);

    uint32_t gen = 5;
    uint32_t pos = 1;  
    for (size_t i = 0; i < slots_half; i++) {
         
        rotation_group_[i] = pos;

        pos *= gen;  
        pos &= (m - 1);
    }

    if (m >= 8) {
        complex_roots_ = std::make_unique<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m)));
        for (size_t i = 0; i < m; i++) {
            root_powers_[i] = complex_roots_->get_root(i);
        }
    } else if (m == 4) {
        root_powers_[0] = {1, 0};
        root_powers_[1] = {0, 1};
        root_powers_[2] = {-1, 0};
        root_powers_[3] = {0, -1};
    }

    cudaMemcpyAsync(gpu_ckks_msg_vec_->twiddle(), root_powers_.data(), m * sizeof(cuDoubleComplex),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(gpu_ckks_msg_vec_->mul_group(), rotation_group_.data(), slots_half * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, s);
}

void PhantomCKKSEncoder::encode_internal(const PhantomContext &context, const cuDoubleComplex *values,
                                         size_t values_size, size_t chain_index, double scale,
                                         PhantomPlaintext &destination, const cudaStream_t &stream) {
    phantom::execution::Scope preparation(phantom::execution::Phase::Preparation);
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (!values && values_size > 0) {
        throw std::invalid_argument("values cannot be null");
    }
    if (values_size > slots_) {
        throw std::invalid_argument("values_size is too large");
    }

    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    if (sparse_slots_ == 0) {
        uint32_t log_sparse_slots = ceil(log2(values_size));
        sparse_slots_ = 1 << log_sparse_slots;
    } else {

    }

    if (sparse_slots_ < 2) {
        throw std::invalid_argument("single value encoding is not available");
    }

    gpu_ckks_msg_vec_->set_sparse_slots(sparse_slots_);
    uint64_t cache_key=0;
    if(phantom::execution::encoding_cache_enabled) {
        cache_key=encoding_hash(values,values_size,chain_index,scale,sparse_slots_);
        auto range=encoding_index.equal_range(cache_key);
        for(auto it=range.first;it!=range.second;++it) {
            auto entry=it->second;
            if(entry->context!=&context || entry->encoder!=this || entry->chain!=chain_index ||
               entry->sparse_slots!=sparse_slots_ || entry->scale!=scale || entry->values.size()!=values_size ||
               std::memcmp(entry->values.data(),values,values_size*sizeof(*values)))continue;
            phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
            destination=entry->plain;encoding_entries.splice(encoding_entries.begin(),encoding_entries,entry);
            ++cache_hits;return;
        }
        ++cache_misses;
    }
    PHANTOM_CHECK_CUDA(cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream));
    auto temp = make_cuda_auto_ptr<cuDoubleComplex>(values_size, stream);
    PHANTOM_CHECK_CUDA(cudaMemsetAsync(temp.get(), 0, values_size * sizeof(cuDoubleComplex), stream));
    PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(temp.get(), values, sizeof(cuDoubleComplex) * values_size, cudaMemcpyHostToDevice, stream));

    uint32_t log_sparse_n = log2(sparse_slots_);
    uint64_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse_and_zero_padding<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            gpu_ckks_msg_vec_->in(), temp.get(), values_size, sparse_slots_, log_sparse_n);

    double fix = scale / static_cast<double>(sparse_slots_);

    special_fft_backward(*gpu_ckks_msg_vec_, fix, stream);

    vector<cuDoubleComplex> temp2(sparse_slots_);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(temp2.data(), gpu_ckks_msg_vec_->in(), sparse_slots_ * sizeof(cuDoubleComplex),
                                       cudaMemcpyDeviceToHost, stream));
     
    cudaStreamSynchronize(stream);

    double max_coeff = 0;
    for (std::size_t i = 0; i < sparse_slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].x));
    }
    for (std::size_t i = 0; i < sparse_slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].y));
    }

    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;

    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
        throw std::invalid_argument("encoded values are too large");
    }

    rns_tool.base_Ql().decompose_array(destination.data(), gpu_ckks_msg_vec_->in(), sparse_slots_ << 1,
                                       (uint32_t) slots_ / sparse_slots_, max_coeff_bit_count, stream);

    nwt_2d_radix8_forward_inplace(destination.data(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    destination.chain_index_ = chain_index;
    destination.scale_ = scale;
    if(phantom::execution::encoding_cache_enabled) {
        const size_t bytes=destination.coeff_count()*sizeof(uint64_t);
        if(bytes<=encoding_limit) {
            phantom::execution::Scope transfer(phantom::execution::Phase::CacheTransfer);
            PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));
            while(encoding_bytes+bytes>encoding_limit)evict_encoding();
            EncodingEntry entry{&context,this,{},chain_index,sparse_slots_,scale,{},cache_key};
            entry.values.assign(values,values+values_size);entry.plain=destination;
            encoding_entries.push_front(std::move(entry));
            encoding_index.emplace(cache_key,encoding_entries.begin());encoding_bytes+=bytes;
        }
    }
}

void PhantomCKKSEncoder::decode_internal(const PhantomContext &context, const PhantomPlaintext &plain,
                                         cuDoubleComplex *destination, const cudaStream_t &stream) {
    if (!destination) {
        throw std::invalid_argument("destination cannot be null");
    }

    auto &context_data = context.get_context_data(plain.chain_index_);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    const size_t coeff_modulus_size = coeff_modulus.size();
    const size_t coeff_count = parms.poly_modulus_degree();
    const size_t rns_poly_uint64_count = coeff_count * coeff_modulus_size;

    if (plain.scale() <= 0 ||
        (static_cast<int>(log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    auto upper_half_threshold = context_data.upper_half_threshold();
    int logn = arith::get_power_of_two(coeff_count);
    auto gpu_upper_half_threshold = make_cuda_auto_ptr<uint64_t>(upper_half_threshold.size(), stream);
    cudaMemcpyAsync(gpu_upper_half_threshold.get(), upper_half_threshold.data(),
                    upper_half_threshold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    gpu_ckks_msg_vec_->set_sparse_slots(sparse_slots_);
    cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream);

    if ((logn < 0) || (coeff_count < POLY_MOD_DEGREE_MIN) || (coeff_count > POLY_MOD_DEGREE_MAX)) {
        throw std::logic_error("invalid parameters");
    }

    double inv_scale = double(1.0) / plain.scale();
     
    auto plain_copy = make_cuda_auto_ptr<uint64_t>(rns_poly_uint64_count, stream);
    cudaMemcpyAsync(plain_copy.get(), plain.data(), rns_poly_uint64_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                    stream);

    nwt_2d_radix8_backward_inplace(plain_copy.get(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    if (decoding_sparse_slots_) {
        rns_tool.base_Ql().compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(),
                                         inv_scale, coeff_count, sparse_slots_ << 1, slots_ / sparse_slots_,
                                         slots_ / decoding_sparse_slots_, stream);
    } else {
        rns_tool.base_Ql().compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(),
                                         inv_scale, coeff_count, sparse_slots_ << 1, slots_ / sparse_slots_, stream);
    }

    special_fft_forward(*gpu_ckks_msg_vec_, stream);

    auto out = make_cuda_auto_ptr<cuDoubleComplex>(sparse_slots_, stream);
    uint32_t log_sparse_n = log2(sparse_slots_);
    size_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            out.get(), gpu_ckks_msg_vec_->in(), sparse_slots_, log_sparse_n);

    if (decoding_sparse_slots_) {
        cudaMemcpyAsync(destination, out.get(), decoding_sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    } else {
        cudaMemcpyAsync(destination, out.get(), sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
}
