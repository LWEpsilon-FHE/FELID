#pragma once

#include <array>
#include <complex>
#include <optional>
#include <vector>

#include "source/fhe_transformer/fhe_adamw.cuh"

namespace fhe_transformer {

enum class CompactMatrixKind { LoraA, LoraB, Wc1, Wc2 };

struct CompactMatrixLayout {
    CompactMatrixKind kind;
    std::vector<int> shape;
    PackingType packing;
    bool complex;
    size_t ciphertexts;
    size_t real_tile_offset;
};

struct CompactOptimizerLayout {
    int slots = 0, batch = 0, row = 0, hidden = 0, rank = 0;
    size_t ciphertexts = 0;
    std::array<CompactMatrixLayout, 14> matrices;
};

inline CompactOptimizerLayout make_compact_optimizer_layout(int slots, int batch,
                                                             int row, int hidden, int rank = 2)
{
    if (slots <= 0 || batch < 2 || (batch & (batch-1)) || row <= 0 ||
        (row & (row-1)) || hidden <= 0 || rank != 2 || slots % (batch*row) ||
        hidden % (slots/(batch*row)) || hidden % row || hidden*batch > slots/2)
        throw std::invalid_argument("Compact optimizer requires integral rank-2 two-layer/binary-classifier packing");
    CompactOptimizerLayout out;
    out.slots=slots; out.batch=batch; out.row=row; out.hidden=hidden; out.rank=rank;
    const size_t real_tiles = static_cast<size_t>(hidden)*rank*batch*row/slots;
    if (real_tiles % 2) throw std::invalid_argument("Compact LoRA tiles must pair into complex lanes");
    size_t offset=0;
    for (int layer=0; layer<2; ++layer) {
        for (int matrix=0; matrix<6; ++matrix) {
            const bool b=matrix>=3;
            out.matrices[layer*6+matrix] = {
                b ? CompactMatrixKind::LoraB : CompactMatrixKind::LoraA,
                b ? std::vector<int>{rank,hidden} : std::vector<int>{hidden,rank},
                b ? PackingType::InterleavedColumnComplex : PackingType::RepeatedColumn,
                b, b ? real_tiles/2 : real_tiles, offset};
            offset += real_tiles;
        }
    }
     
    if (offset/2 > static_cast<size_t>(row*(batch-1)))
        throw std::invalid_argument("LoRA and Wc2 do not fit the compact shared ciphertext");
    const size_t wc1_tiles=static_cast<size_t>(hidden)*hidden*batch/slots;
    if (!wc1_tiles || static_cast<size_t>(hidden)*hidden*batch % slots)
        throw std::invalid_argument("Compact Wc1 tile count is not integral");
    out.matrices[12]={CompactMatrixKind::Wc1,{hidden,hidden},PackingType::RepeatedColumn,
                      false,wc1_tiles,0};
    out.matrices[13]={CompactMatrixKind::Wc2,{hidden,2},PackingType::InterleavedRow,
                      false,1,0};
    out.ciphertexts=1+(wc1_tiles+2*batch-1)/(2*batch);
    return out;
}

struct CompactTileAddress {
    size_t ciphertext;
    int component;
    int source_begin, stride, count, rotation;
};

inline CompactTileAddress compact_tile_address(const CompactOptimizerLayout &layout,
                                               size_t matrix, size_t tile, int component)
{
    const auto &m=layout.matrices.at(matrix);
    if (tile>=m.ciphertexts || component<0 || component>(m.complex || matrix==13 ? 1 : 0))
        throw std::out_of_range("Invalid compact source tile/component");
    if (m.kind==CompactMatrixKind::Wc1) {
        const int lane=static_cast<int>(tile % layout.batch);
        return {1+tile/(2*layout.batch), static_cast<int>((tile/layout.batch)%2),
                0,layout.batch,layout.slots/layout.batch,-lane};
    }
    if (m.kind==CompactMatrixKind::Wc2) {
        const int begin=component*layout.slots/2;
        return {0,component,begin,layout.batch,layout.hidden,begin-(layout.batch-1)};
    }
    const size_t real_tile=m.real_tile_offset+(m.complex ? 2*tile+component : tile);
    const int column=static_cast<int>(real_tile/2);
    const int offset=(column/(layout.batch-1))*layout.batch+column%(layout.batch-1);
    return {0,static_cast<int>(real_tile%2),0,layout.row*layout.batch,
            layout.slots/(layout.row*layout.batch),-offset};
}

inline PhantomCiphertext compact_rotate(FHEAdapterContext &adapter,
                                         const PhantomCiphertext &input, int shift)
{
    if (!shift) return input;
    PhantomCiphertext out;
    adapter.require_evaluator().rotate_vector(const_cast<PhantomCiphertext &>(input),shift,
                                               adapter.require_galois_keys(),out);
    
    return out;
}

inline void compact_add(FHEAdapterContext &adapter, std::optional<PhantomCiphertext> &sum,
                        PhantomCiphertext term)
{
    if (!sum) { sum=std::move(term); return; }
    align_pair_for_add(adapter.require_context(), adapter.level_map, *sum, term, "compact_accumulator", "compact_term");
    assert_scale_close(term,*sum,"compact_term","compact_accumulator");
    term.scale()=sum->scale();
    adapter.require_evaluator().add_inplace(*sum,term);
    
}

struct CompactPackingBuffer {
    std::vector<std::optional<PhantomCiphertext>> lanes;
    std::array<bool,14> appended{};
    double scale=0;
    explicit CompactPackingBuffer(const CompactOptimizerLayout &layout)
        : lanes(2*layout.ciphertexts) {}
};

inline PhantomCiphertext compact_mask_and_rescale(FHEAdapterContext &adapter,
    const PhantomCiphertext &value, const std::vector<std::complex<double>> &mask, double target_scale)
{

    const double prime=static_cast<double>(adapter.require_context().get_context_data(
        value.chain_index()).parms().coeff_modulus().back().value());
    const double plain_scale=target_scale*prime/value.scale();
    if (!std::isfinite(plain_scale) || plain_scale<=0)
        throw std::invalid_argument("Invalid compact mask scale");
    PhantomPlaintext plain;
    encode_values(adapter,mask,value.params_id(),plain_scale,plain);
    return multiply_plain_and_rescale(adapter,value,plain,"compact_mask");
}

inline void compact_pack_append(FHEAdapterContext &adapter, const CompactOptimizerLayout &layout,
                                 CompactPackingBuffer &buffer, size_t matrix,
                                 const FHEBlockTensor &source, bool average_batch)
{
    const auto &m=layout.matrices.at(matrix);
    if (buffer.lanes.size()!=2*layout.ciphertexts || buffer.appended.at(matrix) ||
        source.cts.size()!=m.ciphertexts || source.logical_shape!=m.shape ||
        source.packing!=m.packing || source.is_complex!=m.complex)
        throw std::invalid_argument("Compact source layout/count mismatch or duplicate append");
    buffer.appended[matrix]=true;
    if (buffer.scale==0) buffer.scale=source.cts.front().scale();
    for (size_t tile=0; tile<source.cts.size(); ++tile) {
        auto value=source.cts[tile];
        if (get_level(adapter.level_map,value)<1)
            throw std::runtime_error("Compact packing needs one remaining masking level");
        if (average_batch) {
            for (int shift=1; shift<layout.batch; shift*=2) {
                auto rotated=compact_rotate(adapter,value,shift);
                adapter.require_evaluator().add_inplace(value,rotated);
                
            }
        }
        for (int component=0; component<(m.complex || matrix==13 ? 2 : 1); ++component) {
            const auto address=compact_tile_address(layout,matrix,tile,component);
            auto projected=value;
            std::complex<double> coefficient(average_batch ? 1.0/layout.batch : 1.0,0);
            if (m.complex) {
                PhantomCiphertext conjugate;
                adapter.require_evaluator().complex_conjugate(value,adapter.require_galois_keys(),conjugate);
                if (component) {
                    adapter.require_evaluator().sub_inplace(projected,conjugate);
                    coefficient*=std::complex<double>(0,-0.5);
                } else {
                    adapter.require_evaluator().add_inplace(projected,conjugate);
                    coefficient*=0.5;
                }
                
            }
            std::vector<std::complex<double>> mask(layout.slots,0.0);
            for (int j=0; j<address.count; ++j) {
                int destination=(address.source_begin+j*address.stride-address.rotation)%layout.slots;
                if(destination<0)destination+=layout.slots;
                mask[destination]=coefficient;
            }
            auto positioned=compact_rotate(adapter,projected,address.rotation);
            auto selected=compact_mask_and_rescale(adapter,positioned,mask,buffer.scale);
            compact_add(adapter,buffer.lanes[2*address.ciphertext+address.component],std::move(selected));
        }
    }
}

inline FHEBlockTensor compact_pack_finish(FHEAdapterContext &adapter,
                                           const CompactOptimizerLayout &layout,
                                           CompactPackingBuffer buffer,
                                           const AdamWHEConfig &config,
                                           const std::string &name)
{
    for (bool appended : buffer.appended)
        if (!appended) throw std::invalid_argument("Compact pack is missing a trainable matrix");
    FHEBlockTensor lanes;
    lanes.name=name+"_temporary_real_lanes";
    lanes.logical_shape={static_cast<int>(layout.ciphertexts),2*layout.slots};
    lanes.packing=PackingType::RepeatedRow;

    for (size_t i=0; i<buffer.lanes.size(); ++i) {
        if (!buffer.lanes[i]) {
            const auto &other=*buffer.lanes[i ^ 1];
            auto zero=other;
            adapter.require_evaluator().sub_inplace(zero,const_cast<PhantomCiphertext &>(other));
            
            buffer.lanes[i]=std::move(zero);
        }
    }
    for (size_t i=0; i<buffer.lanes.size(); ++i) {
        FHEBlockTensor one;
        one.cts.push_back(std::move(*buffer.lanes[i]));
        one.name=lanes.name;
        refresh_tensor_metadata(adapter.require_context(),adapter.level_map,one);

        if (one.level==0)
            one=adamw_refresh_to_input_level(adapter,one,config,name+"_exhausted_lane");
        if (std::abs(one.scale-buffer.scale)/buffer.scale>1e-3) {
            std::vector<std::complex<double>> ones(layout.slots,1.0);
            one.cts.front()=compact_mask_and_rescale(adapter,one.cts.front(),ones,buffer.scale);
        }
        lanes.cts.push_back(std::move(one.cts.front()));
    }
    int min_level=adapter.level_map.max_logical_level;
    for (const auto &ct : lanes.cts) min_level=std::min(min_level,get_level(adapter.level_map,ct));
    align_down_to(adapter.require_context(), adapter.level_map, lanes, min_level, lanes.name);
    refresh_tensor_metadata(adapter.require_context(),adapter.level_map,lanes);
    FHEBlockTensor shape;
    shape.name=name; shape.logical_shape=lanes.logical_shape;
    shape.packing=lanes.packing; shape.is_complex=true;
    shape.cts.resize(layout.ciphertexts);
    return pack_independent_complex_lanes(adapter,lanes,shape,name);
}

inline FHEBlockTensor compact_unpack_matrix(FHEAdapterContext &adapter,
                                             const CompactOptimizerLayout &layout,
                                             const FHEBlockTensor &compact, size_t matrix,
                                             const std::string &name)
{
    if (!compact.is_complex || compact.cts.size()!=layout.ciphertexts || compact.level<1)
        throw std::invalid_argument("Invalid compact parameter tensor for unpack");
    const auto &m=layout.matrices.at(matrix);
    FHEBlockTensor real;
    real.name=name+"_temporary_real_tiles"; real.logical_shape=m.shape;
    real.packing=m.packing;
    for (size_t tile=0; tile<m.ciphertexts; ++tile) {
        std::optional<PhantomCiphertext> wc2_sum;
        for (int component=0; component<(m.complex || matrix==13 ? 2 : 1); ++component) {
            const auto address=compact_tile_address(layout,matrix,tile,component);
            auto value=compact.cts.at(address.ciphertext);
            PhantomCiphertext conjugate;
            adapter.require_evaluator().complex_conjugate(value,adapter.require_galois_keys(),conjugate);
            std::complex<double> coefficient(0.5,0);
            if (address.component) {
                adapter.require_evaluator().sub_inplace(value,conjugate);
                coefficient={0,-0.5};
            } else adapter.require_evaluator().add_inplace(value,conjugate);
            
            std::vector<std::complex<double>> mask(layout.slots,0.0);
            for (int j=0; j<address.count; ++j) {
                mask[address.source_begin+j*address.stride]=coefficient;
            }
            auto restored=compact_rotate(adapter,value,-address.rotation);
            restored=compact_mask_and_rescale(adapter,restored,mask,compact.scale);
            const int replicas=(m.kind==CompactMatrixKind::LoraA || m.kind==CompactMatrixKind::LoraB)
                                   ? layout.batch*layout.row : layout.batch;
            for (int shift=1; shift<replicas; shift*=2) {
                auto rotated=compact_rotate(adapter,restored,-shift);
                adapter.require_evaluator().add_inplace(restored,rotated);
                
            }
            if (matrix==13) compact_add(adapter,wc2_sum,std::move(restored));
            else real.cts.push_back(std::move(restored));
        }
        if (matrix==13) real.cts.push_back(std::move(*wc2_sum));
    }
    refresh_tensor_metadata(adapter.require_context(),adapter.level_map,real);
    if (m.complex) {
        FHEBlockTensor shape;
        shape.name=name; shape.logical_shape=m.shape; shape.packing=m.packing;
        shape.is_complex=true; shape.cts.resize(m.ciphertexts);
        return pack_independent_complex_lanes(adapter,real,shape,name);
    }
    real.name=name;
    return real;
}

struct CompactOptimizerState {
    FHEBlockTensor theta;
    std::optional<FHEBlockTensor> m, v;
    int timestep=1;
    std::array<int,5> layout_signature{};
};

inline void compact_optimizer_step(FHEAdapterContext &adapter,
                                    const CompactOptimizerLayout &layout,
                                    CompactOptimizerState &state,
                                    const FHEBlockTensor &gradient,
                                    AdamWHEConfig config)
{
    const std::array<int,5> signature={layout.slots,layout.batch,layout.row,layout.hidden,layout.rank};
    if (state.timestep==1 && state.layout_signature==std::array<int,5>{})
        state.layout_signature=signature;
    if (state.theta.cts.size()!=layout.ciphertexts || gradient.cts.size()!=layout.ciphertexts ||
        state.layout_signature!=signature || state.timestep<=0 ||
        state.m.has_value()!=state.v.has_value() || (state.timestep>1 && !state.m))
        throw std::invalid_argument("Incomplete compact theta/m/v state");
    config.timestep=state.timestep;
    config.keep_intermediate_tensors=false;
    config.keep_moment_tensors=true;
    auto result=adamw_he_update_batched(adapter,state.theta,state.m,state.v,gradient,config,"compact_optimizer");
     
    auto output_config=config;
    output_config.input_level=std::min(config.input_level+1,adapter.level_map.max_logical_level);
    state.theta=adamw_refresh_to_input_level(adapter,result.updated_parameter,output_config,"compact_theta_next_step");
    state.m=std::move(result.m); state.v=std::move(result.v);
    ++state.timestep;
}

}
