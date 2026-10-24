#include "ntt.cuh"
#include "polymath.cuh"
#include "rns.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace phantom {
    DRNSTool::DRNSTool(size_t n, size_t size_P, const RNSBase &base, const std::vector<Modulus> &modulus_QP,
                       const Modulus &t, mul_tech_type mul_tech, const cudaStream_t &stream) {
        size_t base_size = base.size();
        if (base_size < COEFF_MOD_COUNT_MIN || base_size > COEFF_MOD_COUNT_MAX) {
            throw invalid_argument("RNSBase is invalid");
        }

        int log_n = get_power_of_two(n);
        if (log_n < 0 || n > POLY_MOD_DEGREE_MAX || n < POLY_MOD_DEGREE_MIN) {
            throw invalid_argument("poly_modulus_degree is invalid");
        }

        base_.init(base, stream);

        size_t size_QP = modulus_QP.size();
        size_t size_Q = size_QP - size_P;

        mul_tech_ = mul_tech;
        n_ = n;
        size_Q_ = size_Q;
        size_P_ = size_P;
        size_QP_ = size_QP;

        vector<Modulus> modulus_Q(size_Q);
        for (size_t i = 0; i < size_Q; i++)
            modulus_Q[i] = modulus_QP[i];
        RNSBase base_Q(modulus_Q);
        base_Q_.init(base_Q, stream);

        vector<Modulus> modulus_P(size_P);
        for (size_t i = 0; i < size_P; i++)
            modulus_P[i] = modulus_QP[size_Q + i];

        RNSBase base_Ql;
        RNSBase base_QlP;

        if (base_size == size_QP) {  
            base_QlP.init(base);
            base_QlP_.init(base_QlP, stream);
            base_Ql.init(base_QlP.drop(modulus_P));
            base_Ql_.init(base_Ql, stream);
        } else {  
            base_Ql.init(base);
            base_Ql_.init(base_Ql, stream);
            base_QlP.init(base_Ql.extend(RNSBase(modulus_P)));
            base_QlP_.init(base_QlP, stream);
        }

        size_t size_Ql = base_Ql.size();
        size_t size_QlP = size_Ql + size_P;

        std::vector<uint64_t> q_last_mod_q(size_Ql - 1);
        std::vector<uint64_t> q_last_mod_q_shoup(size_Ql - 1);
        std::vector<uint64_t> inv_q_last_mod_q(size_Ql - 1);
        std::vector<uint64_t> inv_q_last_mod_q_shoup(size_Ql - 1);
        for (size_t i = 0; i < size_Ql - 1; i++) {
            q_last_mod_q[i] = barrett_reduce_64(base_Ql[size_Ql - 1].value(), base_Ql[i]);
            q_last_mod_q_shoup[i] = compute_shoup(q_last_mod_q[i], base_Ql[i].value());
            uint64_t value_inv_q_last_mod_q;
            if (!try_invert_uint_mod(base_Ql[size_Ql - 1].value(), base_Ql[i], value_inv_q_last_mod_q)) {
                throw logic_error("invalid rns bases in computing inv_q_last_mod_q");
            }
            inv_q_last_mod_q[i] = value_inv_q_last_mod_q;
            inv_q_last_mod_q_shoup[i] = compute_shoup(value_inv_q_last_mod_q, base_Ql[i].value());
        }
        if (size_Ql > 1) {
            q_last_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql - 1, stream);
            q_last_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql - 1, stream);
            inv_q_last_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql - 1, stream);
            inv_q_last_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql - 1, stream);
            cudaMemcpyAsync(q_last_mod_q_.get(), q_last_mod_q.data(), (size_Ql - 1) * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(q_last_mod_q_shoup_.get(), q_last_mod_q_shoup.data(), (size_Ql - 1) * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(inv_q_last_mod_q_.get(), inv_q_last_mod_q.data(), (size_Ql - 1) * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(inv_q_last_mod_q_shoup_.get(), inv_q_last_mod_q_shoup.data(),
                            (size_Ql - 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        }

        if (size_P != 0) {
            size_t alpha = size_P;

            vector<uint64_t> values_P(size_P);
            for (size_t i = 0; i < size_P; i++)
                values_P[i] = modulus_P[i].value();

            vector<uint64_t> bigP(size_P, 0);
            multiply_many_uint64(values_P.data(), size_P, bigP.data());

            std::vector<uint64_t> bigP_mod_q(size_Ql);
            std::vector<uint64_t> bigP_mod_q_shoup(size_Ql);
            std::vector<uint64_t> bigPInv_mod_q(size_Ql);
            std::vector<uint64_t> bigPInv_mod_q_shoup(size_Ql);
            for (size_t i = 0; i < size_Ql; ++i) {
                auto base_qi = base_Ql.base()[i];
                uint64_t tmp = modulo_uint(bigP.data(), size_P, base_qi);
                bigP_mod_q[i] = tmp;
                bigP_mod_q_shoup[i] = compute_shoup(tmp, base_qi.value());
                if (!try_invert_uint_mod(tmp, base_qi, tmp))
                    throw std::logic_error("invalid rns bases in computing PInv mod q");
                bigPInv_mod_q[i] = tmp;
                bigPInv_mod_q_shoup[i] = compute_shoup(tmp, base_qi.value());
            }

            bigP_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            bigP_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            cudaMemcpyAsync(bigP_mod_q_.get(), bigP_mod_q.data(), size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(bigP_mod_q_shoup_.get(), bigP_mod_q_shoup.data(), size_Ql * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);

            bigPInv_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            bigPInv_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            cudaMemcpyAsync(bigPInv_mod_q_.get(), bigPInv_mod_q.data(), size_Ql * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(bigPInv_mod_q_shoup_.get(), bigPInv_mod_q_shoup.data(), size_Ql * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);

            if (base_size <= size_Q) {
                 
                vector<Modulus> modulus_QlP(size_QlP);
                for (size_t i = 0; i < size_Ql; i++)
                    modulus_QlP[i] = modulus_QP[i];
                for (size_t i = 0; i < size_P; i++)
                    modulus_QlP[size_Ql + i] = modulus_QP[size_Q + i];

                std::vector<uint64_t> partQlHatInv_mod_Ql_concat(size_Ql);
                std::vector<uint64_t> partQlHatInv_mod_Ql_concat_shoup(size_Ql);

                auto beta = static_cast<uint32_t>(ceil(static_cast<double>(size_Ql) / static_cast<double>(alpha)));
                std::vector<std::shared_ptr<BaseConverter>> v_base_part_Ql_to_compl_part_QlP_conv;
                for (size_t beta_idx = 0; beta_idx < beta; beta_idx++) {
                    size_t startPartIdx = alpha * beta_idx;
                    size_t size_PartQl = beta_idx == beta - 1 ? size_Ql - alpha * (beta - 1) : alpha;
                    size_t endPartIdx = startPartIdx + size_PartQl;

                    std::vector<Modulus> modulus_part_Ql{};
                    std::vector<Modulus> modulus_compl_part_QlP = modulus_QlP;

                    for (size_t j = startPartIdx; j < endPartIdx; ++j)
                        modulus_part_Ql.push_back(modulus_QlP[j]);
                    auto first = modulus_compl_part_QlP.cbegin() + startPartIdx;
                    auto last = modulus_compl_part_QlP.cbegin() + endPartIdx;
                    modulus_compl_part_QlP.erase(first, last);

                    auto base_part_Ql = RNSBase(modulus_part_Ql);
                    std::copy_n(base_part_Ql.QHatInvModq(), modulus_part_Ql.size(),
                                partQlHatInv_mod_Ql_concat.begin() + startPartIdx);
                    std::copy_n(base_part_Ql.QHatInvModq_shoup(), modulus_part_Ql.size(),
                                partQlHatInv_mod_Ql_concat_shoup.begin() + startPartIdx);
                    auto base_compl_part_QlP = RNSBase(modulus_compl_part_QlP);

                    auto base_part_Ql_to_compl_part_QlP_conv =
                            make_shared<BaseConverter>(base_part_Ql, base_compl_part_QlP);
                    v_base_part_Ql_to_compl_part_QlP_conv.push_back(base_part_Ql_to_compl_part_QlP_conv);
                }

                partQlHatInv_mod_Ql_concat_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
                partQlHatInv_mod_Ql_concat_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
                cudaMemcpyAsync(partQlHatInv_mod_Ql_concat_.get(), partQlHatInv_mod_Ql_concat.data(),
                                size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(partQlHatInv_mod_Ql_concat_shoup_.get(), partQlHatInv_mod_Ql_concat_shoup.data(),
                                size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

                v_base_part_Ql_to_compl_part_QlP_conv_.resize(beta);
                for (size_t i = 0; i < beta; i++)
                    v_base_part_Ql_to_compl_part_QlP_conv_[i].init(*v_base_part_Ql_to_compl_part_QlP_conv[i], stream);
            }

            BaseConverter base_P_to_Ql_conv(RNSBase(modulus_P), base_Ql);
            base_P_to_Ql_conv_.init(base_P_to_Ql_conv, stream);
        }

        if (!t.is_zero()) {
            t_ = DModulus(t.value(), t.const_ratio().at(0), t.const_ratio().at(1));

            uint64_t q_last_mod_t = barrett_reduce_64(base_Ql.base()[size_Ql - 1].value(), t);
            uint64_t inv_q_last_mod_t;
            if (!try_invert_uint_mod(q_last_mod_t, t, inv_q_last_mod_t))
                throw logic_error("invalid rns bases");

            q_last_mod_t_ = q_last_mod_t;
            inv_q_last_mod_t_ = inv_q_last_mod_t;
            inv_q_last_mod_t_shoup_ = compute_shoup(inv_q_last_mod_t, t.value());
        }

        if (!t.is_zero() && mul_tech == mul_tech_type::none) {
             
            BaseConverter base_q_to_t_conv(base_Ql, RNSBase({t}));
            base_q_to_t_conv_.init(base_q_to_t_conv, stream);

            if (size_P != 0) {
                vector<uint64_t> values_P(size_P);
                for (size_t i = 0; i < size_P; i++)
                    values_P[i] = modulus_P[i].value();

                vector<uint64_t> bigP(size_P, 0);
                multiply_many_uint64(values_P.data(), size_P, bigP.data());

                std::vector<uint64_t> pjInv_mod_q(size_Ql * size_P);
                std::vector<uint64_t> pjInv_mod_q_shoup(size_Ql * size_P);
                for (size_t i = 0; i < size_Ql; i++) {
                    for (size_t j = 0; j < size_P; j++) {
                        uint64_t pj = values_P[j];
                        uint64_t qi = base_Ql.base()[i].value();
                        uint64_t pjInv_mod_qi_value;
                        if (!try_invert_uint_mod(pj, qi, pjInv_mod_qi_value))
                            throw std::logic_error("invalid rns bases when computing pjInv_mod_qi");
                        pjInv_mod_q[i * size_P + j] = pjInv_mod_qi_value;
                        pjInv_mod_q_shoup[i * size_P + j] =
                                compute_shoup(pjInv_mod_qi_value, base_Ql.base()[i].value());
                    }
                }

                pjInv_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql * size_P, stream);
                pjInv_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql * size_P, stream);
                cudaMemcpyAsync(pjInv_mod_q_.get(), pjInv_mod_q.data(), size_Ql * size_P * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(pjInv_mod_q_shoup_.get(), pjInv_mod_q_shoup.data(), size_Ql * size_P * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream);

                std::vector<uint64_t> pjInv_mod_t(size_P);
                std::vector<uint64_t> pjInv_mod_t_shoup(size_P);
                for (size_t j = 0; j < size_P; j++) {
                    uint64_t pjInv_mod_t_value;
                    if (!try_invert_uint_mod(modulus_P[j].value(), t.value(), pjInv_mod_t_value))
                        throw std::logic_error("invalid rns bases when computing pjInv_mod_t");
                    pjInv_mod_t[j] = pjInv_mod_t_value;
                    pjInv_mod_t_shoup[j] = compute_shoup(pjInv_mod_t_value, t.value());
                }

                pjInv_mod_t_ = make_cuda_auto_ptr<uint64_t>(size_P, stream);
                pjInv_mod_t_shoup_ = make_cuda_auto_ptr<uint64_t>(size_P, stream);
                cudaMemcpyAsync(pjInv_mod_t_.get(), pjInv_mod_t.data(), size_P * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(pjInv_mod_t_shoup_.get(), pjInv_mod_t_shoup.data(), size_P * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, stream);

                uint64_t bigP_mod_t_value = modulo_uint(bigP.data(), size_P, t);
                uint64_t bigPInv_mod_t_value;
                if (!try_invert_uint_mod(bigP_mod_t_value, t.value(), bigPInv_mod_t_value))
                    throw std::logic_error("invalid rns bases when computing pjInv_mod_t");
                uint64_t bigPInv_mod_t = bigPInv_mod_t_value;
                uint64_t bigPInv_mod_t_shoup = compute_shoup(bigPInv_mod_t, t.value());

                bigPInv_mod_t_ = bigPInv_mod_t;
                bigPInv_mod_t_shoup_ = bigPInv_mod_t_shoup;

                BaseConverter base_P_to_t_conv(RNSBase(modulus_P), RNSBase({t}));
                base_P_to_t_conv_.init(base_P_to_t_conv, stream);
            }
        }

        if (mul_tech != mul_tech_type::none) {
            vector<uint64_t> values_Ql(size_Ql);
            for (size_t i = 0; i < size_Ql; i++)
                values_Ql[i] = base_Ql.base()[i].value();

            vector<uint64_t> bigQl(size_Ql, 0);
            multiply_many_uint64(values_Ql.data(), size_Ql, bigQl.data());

            uint64_t bigQl_mod_t_value = modulo_uint(bigQl.data(), size_Ql, t);
            uint64_t negQl_mod_t = t.value() - bigQl_mod_t_value;
            uint64_t negQl_mod_t_shoup = compute_shoup(negQl_mod_t, t.value());

            negQl_mod_t_ = negQl_mod_t;
            negQl_mod_t_shoup_ = negQl_mod_t_shoup;

            vector<uint64_t> tInv_mod_q(size_Ql);
            vector<uint64_t> tInv_mod_q_shoup(size_Ql);
            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t tInv_mod_qi_value;
                auto &qi = base_Ql.base()[i];
                if (!try_invert_uint_mod(t.value(), qi.value(), tInv_mod_qi_value))
                    throw std::logic_error("invalid rns bases when computing tInv_mod_qi");
                tInv_mod_q[i] = tInv_mod_qi_value;
                tInv_mod_q_shoup[i] = compute_shoup(tInv_mod_qi_value, qi.value());
            }

            tInv_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            tInv_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            cudaMemcpyAsync(tInv_mod_q_.get(), tInv_mod_q.data(), size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(tInv_mod_q_shoup_.get(), tInv_mod_q_shoup.data(), size_Ql * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
        }

        if (mul_tech == mul_tech_type::behz && base_size <= size_Q) {
            auto primes = get_primes(n_, INTERNAL_MOD_BIT_COUNT, 1);
            auto gamma = primes[0];
            gamma_ = DModulus(gamma.value(), gamma.const_ratio().at(0), gamma.const_ratio().at(1));

            RNSBase base_t_gamma(vector<Modulus>{t, gamma});
            base_t_gamma_.init(base_t_gamma, stream);

            uint64_t temp;
            if (!try_invert_uint_mod(barrett_reduce_64(gamma.value(), t), t, temp)) {
                throw logic_error("invalid rns bases");
            }
            uint64_t inv_gamma_mod_t = temp;
            uint64_t inv_gamma_mod_t_shoup = compute_shoup(inv_gamma_mod_t, t.value());

            inv_gamma_mod_t_ = inv_gamma_mod_t;
            inv_gamma_mod_t_shoup_ = inv_gamma_mod_t_shoup;

            std::vector<uint64_t> prod_t_gamma_mod_q(size_Ql);
            std::vector<uint64_t> prod_t_gamma_mod_q_shoup(size_Ql);
            for (size_t i = 0; i < size_Ql; i++) {
                prod_t_gamma_mod_q[i] = multiply_uint_mod(base_t_gamma[0].value(), base_t_gamma[1].value(), base_Ql[i]);
                prod_t_gamma_mod_q_shoup[i] = compute_shoup(prod_t_gamma_mod_q[i], base_Ql[i].value());
            }

            prod_t_gamma_mod_q_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            prod_t_gamma_mod_q_shoup_ = make_cuda_auto_ptr<uint64_t>(size_Ql, stream);
            cudaMemcpyAsync(prod_t_gamma_mod_q_.get(), prod_t_gamma_mod_q.data(), size_Ql * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(prod_t_gamma_mod_q_shoup_.get(), prod_t_gamma_mod_q_shoup.data(),
                            size_Ql * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

            size_t base_t_gamma_size = 2;
            std::vector<uint64_t> neg_inv_q_mod_t_gamma(base_t_gamma_size);
            std::vector<uint64_t> neg_inv_q_mod_t_gamma_shoup(base_t_gamma_size);
            for (size_t i = 0; i < base_t_gamma_size; i++) {
                auto operand = modulo_uint(base_Ql.big_modulus(), size_Ql, base_t_gamma[i]);
                if (!try_invert_uint_mod(operand, base_t_gamma[i], neg_inv_q_mod_t_gamma[i])) {
                    throw logic_error("invalid rns bases");
                }
                neg_inv_q_mod_t_gamma[i] = negate_uint_mod(neg_inv_q_mod_t_gamma[i], base_t_gamma[i]);
                neg_inv_q_mod_t_gamma_shoup[i] = compute_shoup(neg_inv_q_mod_t_gamma[i], base_t_gamma[i].value());
            }

            neg_inv_q_mod_t_gamma_ = make_cuda_auto_ptr<uint64_t>(base_t_gamma_size, stream);
            neg_inv_q_mod_t_gamma_shoup_ = make_cuda_auto_ptr<uint64_t>(base_t_gamma_size, stream);
            cudaMemcpyAsync(neg_inv_q_mod_t_gamma_.get(), neg_inv_q_mod_t_gamma.data(),
                            base_t_gamma_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(neg_inv_q_mod_t_gamma_shoup_.get(), neg_inv_q_mod_t_gamma_shoup.data(),
                            base_t_gamma_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

            BaseConverter base_q_to_t_gamma_conv(base_Ql, base_t_gamma);
            base_q_to_t_gamma_conv_.init(base_q_to_t_gamma_conv, stream);
        }

        if (mul_tech == mul_tech_type::behz && base_size == size_Q) {

            int total_coeff_bit_count = get_significant_bit_count_uint(base_Q.big_modulus(), base_Q.size());

            size_t base_B_size = size_Q;
            if (32 + t.bit_count() + total_coeff_bit_count >=
                INTERNAL_MOD_BIT_COUNT * static_cast<int>(size_Q) + INTERNAL_MOD_BIT_COUNT) {
                base_B_size++;
            }

            size_t get_primes_count = base_B_size + 1;

            auto baseconv_primes = get_primes(n, INTERNAL_MOD_BIT_COUNT, get_primes_count);
            auto baseconv_primes_iter = baseconv_primes.cbegin();
            Modulus m_sk = *baseconv_primes_iter++;
            m_sk_ = DModulus(m_sk.value(), m_sk.const_ratio().at(0), m_sk.const_ratio().at(1));
            vector<Modulus> base_B_primes;
            copy_n(baseconv_primes_iter, base_B_size, back_inserter(base_B_primes));

            Modulus m_tilde(static_cast<uint64_t>(1) << 32);
            m_tilde_ = DModulus(m_tilde.value(), m_tilde.const_ratio().at(0), m_tilde.const_ratio().at(1));

            RNSBase base_B(base_B_primes);
            base_B_.init(base_B, stream);
            RNSBase base_Bsk(base_B.extend(m_sk));
            base_Bsk_.init(base_Bsk, stream);
            RNSBase base_Bsk_m_tilde(base_Bsk.extend(m_tilde));
            base_Bsk_m_tilde_.init(base_Bsk_m_tilde, stream);

            size_t base_Bsk_size = base_Bsk.size();
            RNSNTT base_Bsk_ntt_tables(log_n, vector(base_Bsk.base(), base_Bsk.base() + base_Bsk_size));
            const auto size_Bsk = base_Bsk_ntt_tables.size();
            gpu_Bsk_tables_.init(n, size_Bsk, stream);
            for (size_t i = 0; i < size_Bsk; i++) {
                auto coeff_modulus = base_Bsk_ntt_tables.get_modulus_at(i);
                auto d_modulus =
                        DModulus(coeff_modulus.value(), coeff_modulus.const_ratio()[0], coeff_modulus.const_ratio()[1]);
                gpu_Bsk_tables_.set(&d_modulus, base_Bsk_ntt_tables.get_ntt_at(i).get_from_root_powers().data(),
                                    base_Bsk_ntt_tables.get_ntt_at(i).get_from_root_powers_shoup().data(),
                                    base_Bsk_ntt_tables.get_ntt_at(i).get_from_inv_root_powers().data(),
                                    base_Bsk_ntt_tables.get_ntt_at(i).get_from_inv_root_powers_shoup().data(),
                                    base_Bsk_ntt_tables.get_ntt_at(i).inv_degree_modulo(),
                                    base_Bsk_ntt_tables.get_ntt_at(i).inv_degree_modulo_shoup(), i, stream);
            }

            std::vector<uint64_t> m_tilde_QHatInvModq(size_Q);
            std::vector<uint64_t> m_tilde_QHatInvModq_shoup(size_Q);
            for (size_t i = 0; i < size_Q; i++) {
                auto qi = base_Q.base()[i];
                auto QHatInvModqi = base_Q.QHatInvModq()[i];
                m_tilde_QHatInvModq[i] = multiply_uint_mod(m_tilde.value(), QHatInvModqi, qi);
                m_tilde_QHatInvModq_shoup[i] = compute_shoup(m_tilde_QHatInvModq[i], qi.value());
            }
            m_tilde_QHatInvModq_ = make_cuda_auto_ptr<uint64_t>(m_tilde_QHatInvModq.size(), stream);
            m_tilde_QHatInvModq_shoup_ = make_cuda_auto_ptr<uint64_t>(m_tilde_QHatInvModq.size(), stream);
            cudaMemcpyAsync(m_tilde_QHatInvModq_.get(), m_tilde_QHatInvModq.data(),
                            sizeof(uint64_t) * m_tilde_QHatInvModq.size(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(m_tilde_QHatInvModq_shoup_.get(), m_tilde_QHatInvModq_shoup.data(),
                            sizeof(uint64_t) * m_tilde_QHatInvModq.size(), cudaMemcpyHostToDevice, stream);

            std::vector<uint64_t> tModBsk(base_Bsk_size);
            std::vector<uint64_t> tModBsk_shoup(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk.size(); i++) {
                tModBsk[i] = t.value();
                tModBsk_shoup[i] = compute_shoup(t.value(), base_Bsk[i].value());
            }
            tModBsk_ = make_cuda_auto_ptr<uint64_t>(tModBsk.size(), stream);
            tModBsk_shoup_ = make_cuda_auto_ptr<uint64_t>(tModBsk_shoup.size(), stream);
            cudaMemcpyAsync(tModBsk_.get(), tModBsk.data(), tModBsk.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(tModBsk_shoup_.get(), tModBsk_shoup.data(), tModBsk_shoup.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);

            BaseConverter base_q_to_Bsk_conv(base_Q, base_Bsk);
            base_q_to_Bsk_conv_.init(base_q_to_Bsk_conv, stream);

            BaseConverter base_q_to_m_tilde_conv(base_Q, RNSBase({m_tilde}));
            base_q_to_m_tilde_conv_.init(base_q_to_m_tilde_conv, stream);

            BaseConverter base_B_to_q_conv(base_B, base_Q);
            base_B_to_q_conv_.init(base_B_to_q_conv, stream);

            BaseConverter base_B_to_m_sk_conv(base_B, RNSBase({m_sk}));
            base_B_to_m_sk_conv_.init(base_B_to_m_sk_conv, stream);

            std::vector<std::uint64_t> prod_B_mod_q(size_Q);
            for (size_t i = 0; i < prod_B_mod_q.size(); i++) {
                prod_B_mod_q[i] = modulo_uint(base_B.big_modulus(), base_B_size, base_Q[i]);
            }
            prod_B_mod_q_ = make_cuda_auto_ptr<uint64_t>(prod_B_mod_q.size(), stream);
            cudaMemcpyAsync(prod_B_mod_q_.get(), prod_B_mod_q.data(), prod_B_mod_q.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);

            uint64_t temp;

            std::vector<uint64_t> inv_prod_q_mod_Bsk(base_Bsk_size);
            std::vector<uint64_t> inv_prod_q_mod_Bsk_shoup(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                temp = modulo_uint(base_Q.big_modulus(), size_Q, base_Bsk[i]);
                if (!try_invert_uint_mod(temp, base_Bsk[i], temp)) {
                    throw logic_error("invalid rns bases");
                }
                inv_prod_q_mod_Bsk[i] = temp;
                inv_prod_q_mod_Bsk_shoup[i] = compute_shoup(temp, base_Bsk[i].value());
            }
            inv_prod_q_mod_Bsk_ = make_cuda_auto_ptr<uint64_t>(inv_prod_q_mod_Bsk.size(), stream);
            inv_prod_q_mod_Bsk_shoup_ = make_cuda_auto_ptr<uint64_t>(inv_prod_q_mod_Bsk_shoup.size(), stream);
            cudaMemcpyAsync(inv_prod_q_mod_Bsk_.get(), inv_prod_q_mod_Bsk.data(),
                            inv_prod_q_mod_Bsk.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(inv_prod_q_mod_Bsk_shoup_.get(), inv_prod_q_mod_Bsk_shoup.data(),
                            inv_prod_q_mod_Bsk_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

            temp = modulo_uint(base_B.big_modulus(), base_B_size, m_sk);
            if (!try_invert_uint_mod(temp, m_sk, temp)) {
                throw logic_error("invalid rns bases");
            }
            uint64_t inv_prod_B_mod_m_sk = temp;
            uint64_t inv_prod_B_mod_m_sk_shoup = compute_shoup(inv_prod_B_mod_m_sk, m_sk.value());
            inv_prod_B_mod_m_sk_ = inv_prod_B_mod_m_sk;
            inv_prod_B_mod_m_sk_shoup_ = inv_prod_B_mod_m_sk_shoup;

            std::vector<uint64_t> inv_m_tilde_mod_Bsk(base_Bsk_size);
            std::vector<uint64_t> inv_m_tilde_mod_Bsk_shoup(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                if (!try_invert_uint_mod(barrett_reduce_64(m_tilde.value(), base_Bsk[i]), base_Bsk[i], temp)) {
                    throw logic_error("invalid rns bases");
                }
                inv_m_tilde_mod_Bsk[i] = temp;
                inv_m_tilde_mod_Bsk_shoup[i] = compute_shoup(temp, base_Bsk[i].value());
            }
            inv_m_tilde_mod_Bsk_ = make_cuda_auto_ptr<uint64_t>(inv_m_tilde_mod_Bsk.size(), stream);
            inv_m_tilde_mod_Bsk_shoup_ = make_cuda_auto_ptr<uint64_t>(inv_m_tilde_mod_Bsk_shoup.size(), stream);
            cudaMemcpyAsync(inv_m_tilde_mod_Bsk_.get(), inv_m_tilde_mod_Bsk.data(),
                            sizeof(uint64_t) * inv_m_tilde_mod_Bsk.size(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(inv_m_tilde_mod_Bsk_shoup_.get(), inv_m_tilde_mod_Bsk_shoup.data(),
                            sizeof(uint64_t) * inv_m_tilde_mod_Bsk_shoup.size(), cudaMemcpyHostToDevice, stream);

            temp = modulo_uint(base_Q.big_modulus(), size_Q, m_tilde);
            if (!try_invert_uint_mod(temp, m_tilde, temp)) {
                throw logic_error("invalid rns bases");
            }
            uint64_t neg_inv_prod_q_mod_m_tilde = negate_uint_mod(temp, m_tilde);
            uint64_t neg_inv_prod_q_mod_m_tilde_shoup = compute_shoup(neg_inv_prod_q_mod_m_tilde, m_tilde.value());
            neg_inv_prod_q_mod_m_tilde_ = neg_inv_prod_q_mod_m_tilde;
            neg_inv_prod_q_mod_m_tilde_shoup_ = neg_inv_prod_q_mod_m_tilde_shoup;

            std::vector<std::uint64_t> prod_q_mod_Bsk(base_Bsk_size);
            for (size_t i = 0; i < base_Bsk_size; i++) {
                prod_q_mod_Bsk[i] = modulo_uint(base_Q.big_modulus(), size_Q, base_Bsk[i]);
            }
            prod_q_mod_Bsk_ = make_cuda_auto_ptr<uint64_t>(prod_q_mod_Bsk.size(), stream);
            cudaMemcpyAsync(prod_q_mod_Bsk_.get(), prod_q_mod_Bsk.data(), prod_q_mod_Bsk.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, stream);
        }

        vector<uint64_t> v_qi(size_Q);
        for (size_t i = 0; i < size_Q; i++) {
            v_qi[i] = modulus_Q[i].value();
        }
        size_t max_q_idx = max_element(v_qi.begin(), v_qi.end()) - v_qi.begin();
        size_t min_q_idx = min_element(v_qi.begin(), v_qi.end()) - v_qi.begin();

        if ((mul_tech == mul_tech_type::hps || mul_tech == mul_tech_type::hps_overq ||
             mul_tech == mul_tech_type::hps_overq_leveled) &&
            (base_size <= size_Q)) {
            size_t qMSB = get_significant_bit_count(v_qi[max_q_idx]);
            size_t sizeQMSB = get_significant_bit_count_uint(&size_Ql, 1);
            size_t tMSB = get_significant_bit_count_uint(t.data(), 1);
            qMSB_ = qMSB;
            sizeQMSB_ = sizeQMSB;
            tMSB_ = tMSB;

            vector<uint64_t> t_QHatInv_mod_q_div_q_mod_t(size_Ql);
            vector<uint64_t> t_QHatInv_mod_q_div_q_mod_t_shoup(size_Ql);
            vector<double> t_QHatInv_mod_q_div_q_frac(size_Ql);
            vector<uint64_t> t_QHatInv_mod_q_B_div_q_mod_t(size_Ql);
            vector<uint64_t> t_QHatInv_mod_q_B_div_q_mod_t_shoup(size_Ql);
            vector<double> t_QHatInv_mod_q_B_div_q_frac(size_Ql);

            for (size_t i = 0; i < size_Ql; i++) {
                auto qi = base_Ql.base()[i];
                auto value_t = t.value();

                std::vector<uint64_t> big_t_QHatInv_mod_qi(2, 0);

                auto qiHatInv_mod_qi = base_Ql.QHatInvModq()[i];

                multiply_uint(&value_t, 1, qiHatInv_mod_qi, 2, big_t_QHatInv_mod_qi.data());

                std::vector<uint64_t> padding_zero_qi(2, 0);
                padding_zero_qi[0] = qi.value();

                std::vector<uint64_t> big_t_QHatInv_mod_q_div_qi(2, 0);

                divide_uint_inplace(big_t_QHatInv_mod_qi.data(), padding_zero_qi.data(), 2,
                                    big_t_QHatInv_mod_q_div_qi.data());

                uint64_t value_t_QHatInv_mod_q_div_q_mod_t = modulo_uint(big_t_QHatInv_mod_q_div_qi.data(), 2, t);

                t_QHatInv_mod_q_div_q_mod_t[i] = value_t_QHatInv_mod_q_div_q_mod_t;
                t_QHatInv_mod_q_div_q_mod_t_shoup[i] = compute_shoup(value_t_QHatInv_mod_q_div_q_mod_t, t.value());
                t_QHatInv_mod_q_div_q_mod_t_ = make_cuda_auto_ptr<uint64_t>(t_QHatInv_mod_q_div_q_mod_t.size(), stream);
                t_QHatInv_mod_q_div_q_mod_t_shoup_ =
                        make_cuda_auto_ptr<uint64_t>(t_QHatInv_mod_q_div_q_mod_t_shoup.size(), stream);
                cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_t_.get(), t_QHatInv_mod_q_div_q_mod_t.data(),
                                t_QHatInv_mod_q_div_q_mod_t.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(t_QHatInv_mod_q_div_q_mod_t_shoup_.get(), t_QHatInv_mod_q_div_q_mod_t_shoup.data(),
                                t_QHatInv_mod_q_div_q_mod_t_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                                stream);

                uint64_t numerator = modulo_uint(big_t_QHatInv_mod_qi.data(), 2, qi);
                uint64_t denominator = qi.value();
                t_QHatInv_mod_q_div_q_frac[i] = static_cast<double>(numerator) / static_cast<double>(denominator);
                t_QHatInv_mod_q_div_q_frac_ = make_cuda_auto_ptr<double>(t_QHatInv_mod_q_div_q_frac.size(), stream);
                cudaMemcpyAsync(t_QHatInv_mod_q_div_q_frac_.get(), t_QHatInv_mod_q_div_q_frac.data(),
                                t_QHatInv_mod_q_div_q_frac.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

                if (qMSB + sizeQMSB >= 52) {
                    size_t qMSBHf = qMSB >> 1;

                    std::vector<uint64_t> QHatInv_mod_qi_B(2, 0);
                    QHatInv_mod_qi_B[0] = qiHatInv_mod_qi;
                    left_shift_uint128(QHatInv_mod_qi_B.data(), qMSBHf, QHatInv_mod_qi_B.data());
                    uint64_t QHatInv_B_mod_qi = modulo_uint(QHatInv_mod_qi_B.data(), 2, qi);

                    std::vector<uint64_t> t_QHatInv_B_mod_qi(2, 0);
                    multiply_uint(&value_t, 1, QHatInv_B_mod_qi, 2, t_QHatInv_B_mod_qi.data());

                    std::vector<uint64_t> t_QHatInv_B_mod_qi_div_qi(2, 0);
                    divide_uint_inplace(t_QHatInv_B_mod_qi.data(), padding_zero_qi.data(), 2,
                                        t_QHatInv_B_mod_qi_div_qi.data());

                    uint64_t value_t_QHatInv_mod_q_B_div_q_mod_t = modulo_uint(t_QHatInv_B_mod_qi_div_qi.data(), 2, t);

                    t_QHatInv_mod_q_B_div_q_mod_t[i] = value_t_QHatInv_mod_q_B_div_q_mod_t;
                    t_QHatInv_mod_q_B_div_q_mod_t_shoup[i] =
                            compute_shoup(value_t_QHatInv_mod_q_B_div_q_mod_t, t.value());
                    t_QHatInv_mod_q_B_div_q_mod_t_ =
                            make_cuda_auto_ptr<uint64_t>(t_QHatInv_mod_q_B_div_q_mod_t.size(), stream);
                    t_QHatInv_mod_q_B_div_q_mod_t_shoup_ =
                            make_cuda_auto_ptr<uint64_t>(t_QHatInv_mod_q_B_div_q_mod_t_shoup.size(), stream);
                    cudaMemcpyAsync(t_QHatInv_mod_q_B_div_q_mod_t_.get(), t_QHatInv_mod_q_B_div_q_mod_t.data(),
                                    t_QHatInv_mod_q_B_div_q_mod_t.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                                    stream);
                    cudaMemcpyAsync(
                            t_QHatInv_mod_q_B_div_q_mod_t_shoup_.get(), t_QHatInv_mod_q_B_div_q_mod_t_shoup.data(),
                            t_QHatInv_mod_q_B_div_q_mod_t_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice,
                            stream);

                    numerator = modulo_uint(t_QHatInv_B_mod_qi.data(), 2, qi);
                    t_QHatInv_mod_q_B_div_q_frac[i] = static_cast<double>(numerator) / static_cast<double>(denominator);
                    t_QHatInv_mod_q_B_div_q_frac_ = make_cuda_auto_ptr<double>(t_QHatInv_mod_q_B_div_q_frac.size(),
                                                                               stream);
                    cudaMemcpyAsync(t_QHatInv_mod_q_B_div_q_frac_.get(), t_QHatInv_mod_q_B_div_q_frac.data(),
                                    t_QHatInv_mod_q_B_div_q_frac.size() * sizeof(double), cudaMemcpyHostToDevice,
                                    stream);
                }
            }
        }

        if (mul_tech == mul_tech_type::hps && base_size == size_Q) {

            size_t size_R = size_Q + 1;
            size_t size_QR = size_Q + size_R;

            auto modulus_R = get_primes_below(n_, modulus_Q[min_q_idx].value(), size_R);
            RNSBase base_Rl(modulus_R);
            base_Rl_.init(base_Rl, stream);
            RNSBase base_QlRl(base_Q.extend(base_Rl));
            base_QlRl_.init(base_QlRl, stream);

            RNSNTT base_QlRl_ntt_tables(log_n, vector(base_QlRl.base(), base_QlRl.base() + size_QR));
            const auto size_QlRl = base_QlRl_ntt_tables.size();
            gpu_QlRl_tables_.init(n, size_QlRl, stream);
            for (size_t i = 0; i < size_QlRl; i++) {
                auto coeff_modulus = base_QlRl_ntt_tables.get_modulus_at(i);
                auto d_modulus =
                        DModulus(coeff_modulus.value(), coeff_modulus.const_ratio()[0], coeff_modulus.const_ratio()[1]);
                gpu_QlRl_tables_.set(&d_modulus, base_QlRl_ntt_tables.get_ntt_at(i).get_from_root_powers().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_root_powers_shoup().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_inv_root_powers().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_inv_root_powers_shoup().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).inv_degree_modulo(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).inv_degree_modulo_shoup(), i, stream);
            }

            auto bigint_Q = base_Q.big_modulus();
            auto bigint_R = base_Rl.big_modulus();

            BaseConverter base_Ql_to_Rl_conv(base_Ql, base_Rl);
            base_Ql_to_Rl_conv_.init(base_Ql_to_Rl_conv, stream);

            BaseConverter base_Rl_to_Ql_conv(base_Rl, base_Q);
            base_Rl_to_Ql_conv_.init(base_Rl_to_Ql_conv, stream);

            vector<double> tRSHatInvModsDivsFrac(size_Q);
            vector<uint64_t> tRSHatInvModsDivsModr(size_R * (size_Q + 1));
            vector<uint64_t> tRSHatInvModsDivsModr_shoup(size_R * (size_Q + 1));

            vector<vector<uint64_t>> tRSHatInvMods(size_QR);
            for (size_t i = 0; i < size_QR; i++) {
                 
                tRSHatInvMods[i].resize(size_R + 2, 0);
                auto SHatInvModsi = base_QlRl.QHatInvModq()[i];
                vector<uint64_t> tR(size_R + 1, 0);
                multiply_uint(bigint_R, size_R, t.value(), size_R + 1, tR.data());
                multiply_uint(tR.data(), size_R + 1, SHatInvModsi, size_R + 2, tRSHatInvMods[i].data());
            }

            for (size_t i = 0; i < size_Q; i++) {
                auto qi = base_Q[i];
                uint64_t tRSHatInvModsModqi = modulo_uint(tRSHatInvMods[i].data(), size_R + 2, qi);
                tRSHatInvModsDivsFrac[i] = static_cast<double>(tRSHatInvModsModqi) / static_cast<double>(qi.value());
            }
            tRSHatInvModsDivsFrac_ = make_cuda_auto_ptr<double>(tRSHatInvModsDivsFrac.size(), stream);
            cudaMemcpyAsync(tRSHatInvModsDivsFrac_.get(), tRSHatInvModsDivsFrac.data(),
                            tRSHatInvModsDivsFrac.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

            vector<vector<uint64_t>> tRSHatInvModsDivs(size_QR);
            for (size_t i = 0; i < size_QR; i++) {
                 
                tRSHatInvModsDivs[i].resize(size_R + 2, 0);
                 
                auto si = base_QlRl.base()[i];
                vector<uint64_t> bigint_si(size_R + 2, 0);
                bigint_si[0] = si.value();
                 
                std::vector<uint64_t> temp_remainder(size_R + 2, 0);
                divide_uint(tRSHatInvMods[i].data(), bigint_si.data(), size_R + 2, tRSHatInvModsDivs[i].data(),
                            temp_remainder.data());
            }

            for (size_t j = 0; j < size_R; j++) {
                auto &rj = modulus_R[j];
                for (size_t i = 0; i < size_Q; i++) {
                     
                    uint64_t tRSHatInvModsDivqiModrj = modulo_uint(tRSHatInvModsDivs[i].data(), size_R + 2, rj);
                    tRSHatInvModsDivsModr[j * (size_Q + 1) + i] = tRSHatInvModsDivqiModrj;
                    tRSHatInvModsDivsModr_shoup[j * (size_Q + 1) + i] =
                            compute_shoup(tRSHatInvModsDivqiModrj, rj.value());
                }
                 
                uint64_t tRSHatInvModsDivrjModrj = modulo_uint(tRSHatInvModsDivs[size_Q + j].data(), size_R + 2, rj);
                tRSHatInvModsDivsModr[j * (size_Q + 1) + size_Q] = tRSHatInvModsDivrjModrj;
                tRSHatInvModsDivsModr_shoup[j * (size_Q + 1) + size_Q] =
                        compute_shoup(tRSHatInvModsDivrjModrj, rj.value());
            }
            tRSHatInvModsDivsModr_ = make_cuda_auto_ptr<uint64_t>(tRSHatInvModsDivsModr.size(), stream);
            tRSHatInvModsDivsModr_shoup_ = make_cuda_auto_ptr<uint64_t>(tRSHatInvModsDivsModr_shoup.size(), stream);
            cudaMemcpyAsync(tRSHatInvModsDivsModr_.get(), tRSHatInvModsDivsModr.data(),
                            tRSHatInvModsDivsModr.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(tRSHatInvModsDivsModr_shoup_.get(), tRSHatInvModsDivsModr_shoup.data(),
                            tRSHatInvModsDivsModr_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        }

        if ((mul_tech == mul_tech_type::hps_overq && base_size == size_Q) ||
            (mul_tech == mul_tech_type::hps_overq_leveled && base_size <= size_Q)) {

            size_t size_Rl = size_Ql;
            size_t size_QlRl = size_Ql + size_Rl;

            auto modulus_Rl = get_primes_below(n, modulus_Q[min_q_idx].value(), size_Rl);
            RNSBase base_Rl(modulus_Rl);
            base_Rl_.init(base_Rl, stream);
            RNSBase base_QlRl(base_Ql.extend(base_Rl));
            base_QlRl_.init(base_QlRl, stream);

            RNSNTT base_QlRl_ntt_tables(log_n, vector(base_QlRl.base(), base_QlRl.base() + size_QlRl));
            gpu_QlRl_tables_.init(n, size_QlRl, stream);
            for (size_t i = 0; i < size_QlRl; i++) {
                auto coeff_modulus = base_QlRl_ntt_tables.get_modulus_at(i);
                auto d_modulus =
                        DModulus(coeff_modulus.value(), coeff_modulus.const_ratio()[0], coeff_modulus.const_ratio()[1]);
                gpu_QlRl_tables_.set(&d_modulus, base_QlRl_ntt_tables.get_ntt_at(i).get_from_root_powers().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_root_powers_shoup().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_inv_root_powers().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).get_from_inv_root_powers_shoup().data(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).inv_degree_modulo(),
                                     base_QlRl_ntt_tables.get_ntt_at(i).inv_degree_modulo_shoup(), i, stream);
            }

            auto bigint_Ql = base_Ql.big_modulus();
            auto bigint_Rl = base_Rl.big_modulus();

            BaseConverter base_Ql_to_Rl_conv(base_Ql, base_Rl);
            base_Ql_to_Rl_conv_.init(base_Ql_to_Rl_conv, stream);

            BaseConverter base_Rl_to_Ql_conv(base_Rl, base_Ql);
            base_Rl_to_Ql_conv_.init(base_Rl_to_Ql_conv, stream);

            vector<double> tQlSlHatInvModsDivsFrac(size_Rl);
            vector<uint64_t> tQlSlHatInvModsDivsModq(size_Ql * (size_Rl + 1));
            vector<uint64_t> tQlSlHatInvModsDivsModq_shoup(size_Ql * (size_Rl + 1));

            vector<vector<uint64_t>> tQlSlHatInvMods(size_QlRl);
            for (size_t i = 0; i < size_QlRl; i++) {
                 
                tQlSlHatInvMods[i].resize(size_Ql + 2, 0);
                auto SHatInvModsi = base_QlRl.QHatInvModq()[i];
                vector<uint64_t> tQl(size_Ql + 1, 0);
                multiply_uint(bigint_Ql, size_Ql, t.value(), size_Ql + 1, tQl.data());
                multiply_uint(tQl.data(), size_Ql + 1, SHatInvModsi, size_Ql + 2, tQlSlHatInvMods[i].data());
            }

            for (size_t j = 0; j < size_Rl; j++) {
                auto rj = base_Rl.base()[j];
                uint64_t tQlSlHatInvModsModrj = modulo_uint(tQlSlHatInvMods[size_Ql + j].data(), size_Ql + 2, rj);
                tQlSlHatInvModsDivsFrac[j] =
                        static_cast<double>(tQlSlHatInvModsModrj) / static_cast<double>(rj.value());
            }
            tQlSlHatInvModsDivsFrac_ = make_cuda_auto_ptr<double>(tQlSlHatInvModsDivsFrac.size(), stream);
            cudaMemcpyAsync(tQlSlHatInvModsDivsFrac_.get(), tQlSlHatInvModsDivsFrac.data(),
                            tQlSlHatInvModsDivsFrac.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

            vector<vector<uint64_t>> tQlSlHatInvModsDivs(size_QlRl);
            for (size_t i = 0; i < size_QlRl; i++) {
                 
                tQlSlHatInvModsDivs[i].resize(size_Ql + 2, 0);
                 
                auto si = base_QlRl.base()[i];
                vector<uint64_t> bigint_si(size_Ql + 2, 0);
                bigint_si[0] = si.value();
                 
                std::vector<uint64_t> temp_remainder(size_Ql + 2, 0);
                divide_uint(tQlSlHatInvMods[i].data(), bigint_si.data(), size_Ql + 2, tQlSlHatInvModsDivs[i].data(),
                            temp_remainder.data());
            }

            for (size_t i = 0; i < size_Ql; i++) {
                auto &qi = base_Ql.base()[i];
                for (size_t j = 0; j < size_Rl; j++) {
                     
                    uint64_t tQlSlHatInvModsDivrjModqi =
                            modulo_uint(tQlSlHatInvModsDivs[size_Ql + j].data(), size_Ql + 2, qi);
                    tQlSlHatInvModsDivsModq[i * (size_Rl + 1) + j] = tQlSlHatInvModsDivrjModqi;
                    tQlSlHatInvModsDivsModq_shoup[i * (size_Rl + 1) + j] =
                            compute_shoup(tQlSlHatInvModsDivrjModqi, qi.value());
                }
                 
                uint64_t tQlSlHatInvModsDivqiModqi = modulo_uint(tQlSlHatInvModsDivs[i].data(), size_Ql + 2, qi);
                tQlSlHatInvModsDivsModq[i * (size_Rl + 1) + size_Rl] = tQlSlHatInvModsDivqiModqi;
                tQlSlHatInvModsDivsModq_shoup[i * (size_Rl + 1) + size_Rl] =
                        compute_shoup(tQlSlHatInvModsDivqiModqi, qi.value());
            }
            tQlSlHatInvModsDivsModq_ = make_cuda_auto_ptr<uint64_t>(tQlSlHatInvModsDivsModq.size(), stream);
            tQlSlHatInvModsDivsModq_shoup_ = make_cuda_auto_ptr<uint64_t>(tQlSlHatInvModsDivsModq_shoup.size(), stream);
            cudaMemcpyAsync(tQlSlHatInvModsDivsModq_.get(), tQlSlHatInvModsDivsModq.data(),
                            tQlSlHatInvModsDivsModq.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(tQlSlHatInvModsDivsModq_shoup_.get(), tQlSlHatInvModsDivsModq_shoup.data(),
                            tQlSlHatInvModsDivsModq_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

            if (mul_tech == mul_tech_type::hps_overq_leveled && base_size < size_Q) {
                 
                if (size_Q - size_Ql < 1)
                    throw std::logic_error("Something is wrong, check rnstool.");
                size_t size_QlDrop = size_Q - size_Ql;

                vector<Modulus> modulus_QlDrop(size_QlDrop);
                for (size_t i = 0; i < size_QlDrop; i++)
                    modulus_QlDrop[i] = modulus_Q[size_Ql + i];
                RNSBase base_QlDrop(modulus_QlDrop);
                base_QlDrop_.init(base_QlDrop, stream);

                BaseConverter base_Q_to_Rl_conv(base_Q, base_Rl);
                base_Q_to_Rl_conv_.init(base_Q_to_Rl_conv, stream);

                BaseConverter base_Ql_to_QlDrop_conv(base_Ql, base_QlDrop);
                base_Ql_to_QlDrop_conv_.init(base_Ql_to_QlDrop_conv, stream);

                vector<double> QlQHatInvModqDivqFrac(size_QlDrop);
                vector<uint64_t> QlQHatInvModqDivqModq(size_Ql * (size_QlDrop + 1));
                vector<uint64_t> QlQHatInvModqDivqModq_shoup(size_Ql * (size_QlDrop + 1));

                vector<vector<uint64_t>> QlQHatInvModq(size_Q);
                for (size_t i = 0; i < size_Q; i++) {
                     
                    QlQHatInvModq[i].resize(size_Ql + 1, 0);
                    multiply_uint(bigint_Ql, size_Ql, base_Q.QHatInvModq()[i], size_Ql + 1, QlQHatInvModq[i].data());
                }

                for (size_t j = 0; j < size_QlDrop; j++) {
                    auto rj = base_QlDrop.base()[j];
                    uint64_t QlQHatInvModqModrj = modulo_uint(QlQHatInvModq[size_Ql + j].data(), size_Ql + 1, rj);
                    QlQHatInvModqDivqFrac[j] =
                            static_cast<double>(QlQHatInvModqModrj) / static_cast<double>(rj.value());
                }
                QlQHatInvModqDivqFrac_ = make_cuda_auto_ptr<double>(QlQHatInvModqDivqFrac.size(), stream);
                cudaMemcpyAsync(QlQHatInvModqDivqFrac_.get(), QlQHatInvModqDivqFrac.data(),
                                QlQHatInvModqDivqFrac.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

                vector<vector<uint64_t>> QlQHatInvModqDivq(size_Q);
                for (size_t i = 0; i < size_Q; i++) {
                     
                    QlQHatInvModqDivq[i].resize(size_Ql + 1, 0);
                     
                    auto qi = base_Q.base()[i];
                    vector<uint64_t> bigint_qi(size_Ql + 1, 0);
                    bigint_qi[0] = qi.value();
                     
                    std::vector<uint64_t> temp_remainder(size_Ql + 1, 0);
                    divide_uint(QlQHatInvModq[i].data(), bigint_qi.data(), size_Ql + 1, QlQHatInvModqDivq[i].data(),
                                temp_remainder.data());
                }

                for (size_t i = 0; i < size_Ql; i++) {
                    auto &qi = base_Ql.base()[i];
                    for (size_t j = 0; j < size_QlDrop; j++) {
                         
                        uint64_t QlQHatInvModqDivrjModqi =
                                modulo_uint(QlQHatInvModqDivq[size_Ql + j].data(), size_Ql + 1, qi);
                        QlQHatInvModqDivqModq[i * (size_QlDrop + 1) + j] = QlQHatInvModqDivrjModqi;
                        QlQHatInvModqDivqModq_shoup[i * (size_QlDrop + 1) + j] =
                                compute_shoup(QlQHatInvModqDivrjModqi, qi.value());
                    }
                     
                    uint64_t QlQHatInvModqDivqiModqi = modulo_uint(QlQHatInvModqDivq[i].data(), size_Ql + 1, qi);
                    QlQHatInvModqDivqModq[i * (size_QlDrop + 1) + size_QlDrop] = QlQHatInvModqDivqiModqi;
                    QlQHatInvModqDivqModq_shoup[i * (size_QlDrop + 1) + size_QlDrop] =
                            compute_shoup(QlQHatInvModqDivqiModqi, qi.value());
                }
                QlQHatInvModqDivqModq_ = make_cuda_auto_ptr<uint64_t>(QlQHatInvModqDivqModq.size(), stream);
                QlQHatInvModqDivqModq_shoup_ = make_cuda_auto_ptr<uint64_t>(QlQHatInvModqDivqModq_shoup.size(), stream);
                cudaMemcpyAsync(QlQHatInvModqDivqModq_.get(), QlQHatInvModqDivqModq.data(),
                                QlQHatInvModqDivqModq.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(QlQHatInvModqDivqModq_shoup_.get(), QlQHatInvModqDivqModq_shoup.data(),
                                QlQHatInvModqDivqModq_shoup.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
            }
        }
    }

    __global__ void perform_final_multiplication(uint64_t *dst, const uint64_t *src, const uint64_t inv_gamma_mod_t,
                                                 const uint64_t inv_gamma_mod_t_shoup, const uint64_t poly_degree,
                                                 const DModulus *base_t_gamma) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree; tid += blockDim.x * gridDim.x) {
            DModulus t = base_t_gamma[0];
            uint64_t gamma_value = base_t_gamma[1].value();
            uint64_t threshold_value = gamma_value >> 1;
            uint64_t temp;
             
            if (src[poly_degree + tid] > threshold_value) {
                 
                temp = barrett_reduce_uint64_uint64((gamma_value - src[poly_degree + tid]), t.value(),
                                                    t.const_ratio()[1]);
                temp = add_uint64_uint64_mod(src[tid], temp, t.value());
            } else {
                 
                temp = barrett_reduce_uint64_uint64(src[poly_degree + tid], t.value(), t.const_ratio()[1]);
                temp = sub_uint64_uint64_mod(src[tid], temp, t.value());
            }
             
            dst[tid] = multiply_and_reduce_shoup(temp, inv_gamma_mod_t, inv_gamma_mod_t_shoup, t.value());
        }
    }

    void DRNSTool::behz_decrypt_scale_and_round(uint64_t *src, uint64_t *temp, const DNTTTable &rns_table,
                                                uint64_t temp_mod_size, uint64_t poly_modulus_degree,
                                                uint64_t *dst, const cudaStream_t &stream) const {
        size_t base_q_size = base_Ql_.size();
        size_t base_t_gamma_size = base_t_gamma_.size();
        size_t coeff_mod_size = rns_table.size();

        uint64_t gridDimGlb;
        gridDimGlb = n_ * base_q_size / blockDimGlb.x;
        multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                src, prod_t_gamma_mod_q(), prod_t_gamma_mod_q_shoup(),
                base_Ql_.base(), temp, n_, base_q_size);

        if (temp_mod_size >= base_t_gamma_size) {
             
            base_q_to_t_gamma_conv_.bConv_BEHZ(temp, temp, n_, stream);

            if (coeff_mod_size >= base_t_gamma_size) {
                gridDimGlb = n_ * base_t_gamma_size / blockDimGlb.x;
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        temp, neg_inv_q_mod_t_gamma(), neg_inv_q_mod_t_gamma_shoup(),
                        base_t_gamma_.base(), temp, n_, base_t_gamma_size);
            } else {
                 
                gridDimGlb = n_ * coeff_mod_size / blockDimGlb.x;
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        temp, neg_inv_q_mod_t_gamma(), neg_inv_q_mod_t_gamma_shoup(),
                        base_t_gamma_.base(), temp, n_, coeff_mod_size);
            }

            gridDimGlb = n_ / blockDimGlb.x;
            perform_final_multiplication<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, temp, inv_gamma_mod_t_, inv_gamma_mod_t_shoup_,
                    n_, base_t_gamma_.base());
        } else {
             
            auto t_gamma = make_cuda_auto_ptr<uint64_t>(base_t_gamma_size * n_, stream);

            base_q_to_t_gamma_conv_.bConv_BEHZ(t_gamma.get(), temp, n_, stream);

            if (coeff_mod_size >= base_t_gamma_size) {
                gridDimGlb = n_ * base_t_gamma_size / blockDimGlb.x;
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        t_gamma.get(), neg_inv_q_mod_t_gamma(), neg_inv_q_mod_t_gamma_shoup(), base_t_gamma_.base(),
                        t_gamma.get(), n_, base_t_gamma_size);
            } else {
                gridDimGlb = n_ * coeff_mod_size / blockDimGlb.x;
                multiply_scalar_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        t_gamma.get(), neg_inv_q_mod_t_gamma(), neg_inv_q_mod_t_gamma_shoup(), base_t_gamma_.base(),
                        t_gamma.get(), n_, coeff_mod_size);
            }

            gridDimGlb = n_ / blockDimGlb.x;
            perform_final_multiplication<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst, t_gamma.get(), inv_gamma_mod_t_, inv_gamma_mod_t_shoup_,
                    n_, base_t_gamma_.base());
        }
    }

    __global__ void divide_and_round_q_last_kernel(uint64_t *dst, const uint64_t *src, const DModulus *base_q,
                                                   const uint64_t *inv_q_last_mod_q,
                                                   const uint64_t *inv_q_last_mod_q_shoup, const uint64_t poly_degree,
                                                   const uint64_t next_base_q_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * next_base_q_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / poly_degree;
            DModulus mod = base_q[twr];
            uint64_t q_last_inv = inv_q_last_mod_q[twr];
            uint64_t q_last_inv_shoup = inv_q_last_mod_q_shoup[twr];

            uint64_t c_last_coeff = src[(tid % poly_degree) + next_base_q_size * poly_degree];

            uint64_t temp;

            temp = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);
            temp = sub_uint64_uint64_mod(src[tid], temp, mod.value());

            dst[tid] = multiply_and_reduce_shoup(temp, q_last_inv, q_last_inv_shoup, mod.value());
        }
    }

    void DRNSTool::divide_and_round_q_last(const uint64_t *src, size_t cipher_size, uint64_t *dst,
                                           const cudaStream_t &stream) const {
        size_t size_Ql = base_Ql_.size();
        size_t next_size_Ql = size_Ql - 1;

        uint64_t gridDimGlb = n_ * next_size_Ql / blockDimGlb.x;
        for (size_t i = 0; i < cipher_size; i++) {
            divide_and_round_q_last_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    dst + i * next_size_Ql * n_, src + i * size_Ql * n_, base_Ql_.base(), inv_q_last_mod_q(),
                    inv_q_last_mod_q_shoup(), n_, next_size_Ql);
        }
    }

    __global__ void divide_and_round_reduce_q_last_kernel(uint64_t *dst, const uint64_t *src, const DModulus *base_q,
                                                          const uint64_t poly_degree, const uint64_t next_base_q_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * next_base_q_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / poly_degree;
            DModulus mod = base_q[twr];
            uint64_t c_last_coeff = src[(tid % poly_degree) + next_base_q_size * poly_degree];

            dst[tid] = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);
        }
    }

    __global__ void divide_and_round_ntt_inv_scalar_kernel(uint64_t *dst, const uint64_t *src, const DModulus *base_q,
                                                           const uint64_t *inv_q_last_mod_q,
                                                           const uint64_t *inv_q_last_mod_q_shoup,
                                                           const uint64_t poly_degree,
                                                           const uint64_t next_base_q_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * next_base_q_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / poly_degree;
            uint64_t mod_value = base_q[twr].value();
            uint64_t q_last_inv = inv_q_last_mod_q[twr];
            uint64_t q_last_inv_shoup = inv_q_last_mod_q_shoup[twr];

            uint64_t temp;

            temp = sub_uint64_uint64_mod(src[tid], dst[tid], mod_value);
            dst[tid] = multiply_and_reduce_shoup(temp, q_last_inv, q_last_inv_shoup, mod_value);
        }
    }

    void DRNSTool::divide_and_round_q_last_ntt(uint64_t *src, size_t cipher_size, const DNTTTable &rns_tables,
                                               uint64_t *dst, const cudaStream_t &stream) const {
        size_t base_q_size = base_Ql_.size();
        auto next_base_q_size = base_q_size - 1;
        uint64_t gridDimGlb = n_ * next_base_q_size / blockDimGlb.x;

        for (size_t i = 0; i < cipher_size; i++) {
            uint64_t *ci_in = src + i * n_ * base_q_size;
            uint64_t *ci_out = dst + i * n_ * next_base_q_size;

            nwt_2d_radix8_backward_inplace(ci_in, rns_tables, 1, base_q_size - 1, stream);

            divide_and_round_reduce_q_last_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    ci_out, ci_in, base_Ql_.base(), n_, next_base_q_size);

            nwt_2d_radix8_forward_inplace(ci_out, rns_tables, next_base_q_size, 0, stream);

            divide_and_round_ntt_inv_scalar_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    ci_out, ci_in, base_Ql_.base(), inv_q_last_mod_q(), inv_q_last_mod_q_shoup(), n_, next_base_q_size);
        }
    }

    __global__ static void bgv_mod_t_divide_q_kernel(
            uint64_t *dst, const uint64_t *cx, const uint64_t *ci_last,
            const uint64_t *q_last_mod_qi, const uint64_t *q_last_mod_qi_shoup,
            const uint64_t *inv_q_last_mod_q, const uint64_t *inv_q_last_mod_q_shoup,
            const uint64_t inv_q_last_mod_t, const uint64_t inv_q_last_mod_t_shoup,
            const DModulus *base_Ql, size_t next_base_q_size,
            uint64_t t, uint64_t t_mu_hi, uint64_t n) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
             tid < n * next_base_q_size; tid += blockDim.x * gridDim.x) {
            size_t i = tid / n;
            auto qi = base_Ql[i].value();
            auto qi_mu_hi = base_Ql[i].const_ratio()[1];
            uint64_t ci_last_value = ci_last[tid % n];
            uint64_t delta = barrett_reduce_uint64_uint64(ci_last_value, qi, qi_mu_hi);
            uint64_t ci_last_mod_t = barrett_reduce_uint64_uint64(ci_last_value, t, t_mu_hi);
            uint64_t temp = multiply_and_reduce_shoup(ci_last_mod_t, inv_q_last_mod_t, inv_q_last_mod_t_shoup, t);
            uint64_t correction = multiply_and_reduce_shoup(temp, q_last_mod_qi[i], q_last_mod_qi_shoup[i], qi);
            temp = sub_uint64_uint64_mod(cx[tid], delta, qi);
            temp = add_uint64_uint64_mod(temp, correction, qi);
            dst[tid] = multiply_and_reduce_shoup(temp, inv_q_last_mod_q[i], inv_q_last_mod_q_shoup[i], qi);
        }
    }

    void DRNSTool::mod_t_and_divide_q_last_ntt(uint64_t *src, size_t cipher_size, const DNTTTable &rns_tables,
                                               uint64_t *dst, const cudaStream_t &stream) const {
        size_t base_q_size = base_Ql_.size();
        auto next_base_q_size = base_q_size - 1;

        for (size_t i = 0; i < cipher_size; i++) {
            uint64_t *ci_in = src + i * n_ * base_q_size;
            uint64_t *ci_out = dst + i * n_ * next_base_q_size;
            uint64_t *ci_last = ci_in + n_ * next_base_q_size;

            nwt_2d_radix8_backward_inplace(ci_in, rns_tables, base_q_size, 0, stream);

            uint64_t gridDimGlb = n_ * next_base_q_size / blockDimGlb.x;
            bgv_mod_t_divide_q_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                    ci_out, ci_in, ci_last,
                    q_last_mod_q(), q_last_mod_q_shoup(),
                    inv_q_last_mod_q(), inv_q_last_mod_q_shoup(),
                    inv_q_last_mod_t(), inv_q_last_mod_t_shoup(),
                    rns_tables.modulus(), next_base_q_size,
                    t_.value(), t_.const_ratio()[1], n_);

            nwt_2d_radix8_forward_inplace(ci_out, rns_tables, next_base_q_size, 0, stream);
        }
    }

    void DRNSTool::decrypt_mod_t(uint64_t *dst, const uint64_t *src, const uint64_t poly_degree,
                                 const cudaStream_t &stream) const {
        base_q_to_t_conv_.exact_convert_array(dst, src, poly_degree, stream);
    }

    void DRNSTool::fastbconv_m_tilde(uint64_t *dst, uint64_t *src, const cudaStream_t &stream) const {
        size_t base_Q_size = base_Ql_.size();
        size_t base_Bsk_size = base_Bsk_.size();
        auto n = n_;

        auto temp_bconv = make_cuda_auto_ptr<uint64_t>(base_Q_size * n, stream);

        constexpr int unroll_factor = 2;

        uint64_t gridDimGlb = base_Q_size * n / unroll_factor / blockDimGlb.x;
        bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                temp_bconv.get(), src, m_tilde_QHatInvModq(), m_tilde_QHatInvModq_shoup(),
                base_Q_.base(), base_Q_size, n);

        gridDimGlb = base_Bsk_size * n / unroll_factor / blockDimGlb.x;
        bconv_matmul_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * base_Bsk_size * base_Q_size, stream>>>(
                dst, temp_bconv.get(), base_q_to_Bsk_conv_.QHatModp(), base_Q_.base(), base_Q_size, base_Bsk_.base(),
                base_Bsk_size, n);

        gridDimGlb = 1 * n / unroll_factor / blockDimGlb.x;  
        bconv_matmul_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * 1 * base_Q_size, stream>>>(
                dst + n * base_Bsk_size, temp_bconv.get(), base_q_to_m_tilde_conv_.QHatModp(), base_Q_.base(),
                base_Q_size, base_q_to_m_tilde_conv_.obase().base(), 1, n);
    }

    __global__ void sm_mrq_kernel(uint64_t *dst, const uint64_t *src, const uint64_t m_tilde,
                                  const uint64_t neg_inv_prod_q_mod_m_tilde,
                                  const uint64_t neg_inv_prod_q_mod_m_tilde_shoup, const DModulus *base_Bsk,
                                  const uint64_t *prod_q_mod_Bsk, const uint64_t *inv_m_tilde_mod_Bsk,
                                  const uint64_t *inv_m_tilde_mod_Bsk_shoup, const uint64_t poly_degree,
                                  const uint64_t base_Bsk_size) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < poly_degree * base_Bsk_size;
             tid += blockDim.x * gridDim.x) {
            size_t twr = tid / poly_degree;
            DModulus Bski = base_Bsk[twr];
            uint64_t prod_q_mod_Bski = prod_q_mod_Bsk[twr];
            uint64_t inv_m_tilde_mod_Bski = inv_m_tilde_mod_Bsk[twr];
            uint64_t inv_m_tilde_mod_Bski_shoup = inv_m_tilde_mod_Bsk_shoup[twr];

            uint64_t r_m_tilde = src[(tid % poly_degree) + base_Bsk_size * poly_degree];
             
            r_m_tilde = multiply_and_reduce_shoup(r_m_tilde, neg_inv_prod_q_mod_m_tilde,
                                                  neg_inv_prod_q_mod_m_tilde_shoup, m_tilde);
             
            if (r_m_tilde >= m_tilde >> 1) {
                r_m_tilde += Bski.value() - m_tilde;
            }
             
            uint128_t temp;
            temp = multiply_uint64_uint64(r_m_tilde, prod_q_mod_Bski);
            temp = add_uint128_uint64(temp, src[tid]);
            temp.lo = barrett_reduce_uint128_uint64(temp, Bski.value(), Bski.const_ratio());
            dst[tid] =
                    multiply_and_reduce_shoup(temp.lo, inv_m_tilde_mod_Bski, inv_m_tilde_mod_Bski_shoup, Bski.value());
        }
    }

    void DRNSTool::sm_mrq(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        size_t base_Bsk_size = base_Bsk_.size();

        uint64_t gridDimGlb = n_ * base_Bsk_size / blockDimGlb.x;
        sm_mrq_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, m_tilde_.value(),
                neg_inv_prod_q_mod_m_tilde_,  
                neg_inv_prod_q_mod_m_tilde_shoup_,
                base_Bsk_.base(),  
                prod_q_mod_Bsk(),  
                inv_m_tilde_mod_Bsk(),  
                inv_m_tilde_mod_Bsk_shoup(),  
                n_, base_Bsk_size);
    }

    __global__ static void
    bconv_fuse_sub_mul_unroll2_kernel(uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *input_base_Bsk,
                                      const uint64_t *inv_prod_q_mod_Bsk, const uint64_t *inv_prod_q_mod_Bsk_shoup,
                                      const uint64_t *QHatModp, const DModulus *ibase, uint64_t ibase_size,
                                      const DModulus *obase, uint64_t obase_size, uint64_t n) {
        constexpr int unroll_number = 2;
        extern __shared__ uint64_t s_QHatModp[];
        for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
            s_QHatModp[idx] = QHatModp[idx];
        }
        __syncthreads();

        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n / unroll_number;
             tid += blockDim.x * gridDim.x) {
            const size_t degree_idx = unroll_number * (tid / obase_size);
            const size_t out_prime_idx = tid % obase_size;

            uint128_t2 accum =
                    base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_QHatModp, out_prime_idx, n, ibase_size, degree_idx);

            uint64_t obase_value = obase[out_prime_idx].value();
            uint64_t obase_ratio[2] = {obase[out_prime_idx].const_ratio()[0], obase[out_prime_idx].const_ratio()[1]};
            auto &scale = inv_prod_q_mod_Bsk[out_prime_idx];
            auto &scale_shoup = inv_prod_q_mod_Bsk_shoup[out_prime_idx];
            uint64_t out1, out2;
            uint64_t input1, input2;

            out1 = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
            out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
            ld_two_uint64(input1, input2, input_base_Bsk + out_prime_idx * n + degree_idx);

            sub_uint64_uint64(obase_value, out1, out1);
            add_uint64_uint64(input1, out1, out1);
            out1 = multiply_and_reduce_shoup(out1, scale, scale_shoup, obase_value);

            sub_uint64_uint64(obase_value, out2, out2);
            add_uint64_uint64(input2, out2, out2);
            out2 = multiply_and_reduce_shoup(out2, scale, scale_shoup, obase_value);

            st_two_uint64(dst + out_prime_idx * n + degree_idx, out1, out2);
        }
    }

    void DRNSTool::fast_floor(uint64_t *input_base_q, uint64_t *input_base_Bsk, uint64_t *out_base_Bsk,
                              const cudaStream_t &stream) const {
        size_t base_Bsk_size = base_Bsk_.size();
        size_t base_Q_size = base_Q_.size();
        auto n = n_;

        auto temp_bconv = make_cuda_auto_ptr<uint64_t>(base_Q_size * n, stream);

        constexpr int unroll_factor = 2;

        uint64_t gridDimGlb = base_Q_size * n / unroll_factor / blockDimGlb.x;
        bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                temp_bconv.get(), input_base_q, base_Q_.QHatInvModq(),
                base_Q_.QHatInvModq_shoup(), base_Q_.base(), base_Q_size,
                n);

        gridDimGlb = base_Bsk_size * n / unroll_factor / blockDimGlb.x;
        bconv_fuse_sub_mul_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * base_Bsk_size * base_Q_size, stream>>>(
                out_base_Bsk, temp_bconv.get(), input_base_Bsk, inv_prod_q_mod_Bsk(), inv_prod_q_mod_Bsk_shoup(),
                base_q_to_Bsk_conv_.QHatModp(), base_Q_.base(), base_Q_size, base_Bsk_.base(), base_Bsk_size, n);
    }

    __global__ static void bconv_fuse_sub_mul_single_unroll2_kernel(
            uint64_t *dst, const uint64_t *xi_qiHatInv_mod_qi, const uint64_t *input_base_Bsk,
            uint64_t inv_prod_q_mod_Bsk, uint64_t inv_prod_q_mod_Bsk_shoup, const uint64_t *QHatModp,
            const DModulus *ibase, uint64_t ibase_size, DModulus obase, uint64_t obase_size, uint64_t n) {
        constexpr const int unroll_number = 2;
        extern __shared__ uint64_t s_QHatModp[];
        for (size_t idx = threadIdx.x; idx < obase_size * ibase_size; idx += blockDim.x) {
            s_QHatModp[idx] = QHatModp[idx];
        }
        __syncthreads();

        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < obase_size * n / unroll_number;
             tid += blockDim.x * gridDim.x) {
            const size_t degree_idx = unroll_number * (tid / obase_size);
            const size_t out_prime_idx = tid % obase_size;

            uint128_t2 accum =
                    base_convert_acc_unroll2(xi_qiHatInv_mod_qi, s_QHatModp, out_prime_idx, n, ibase_size, degree_idx);

            uint64_t obase_value = obase.value();
            uint64_t obase_ratio[2] = {obase.const_ratio()[0], obase.const_ratio()[1]};
            auto &scale = inv_prod_q_mod_Bsk;
            auto &scale_shoup = inv_prod_q_mod_Bsk_shoup;
            uint64_t out1, out2;
            uint64_t input1, input2;

            out1 = barrett_reduce_uint128_uint64(accum.x, obase_value, obase_ratio);
            out2 = barrett_reduce_uint128_uint64(accum.y, obase_value, obase_ratio);
            ld_two_uint64(input1, input2, input_base_Bsk + out_prime_idx * n + degree_idx);

            sub_uint64_uint64(obase_value, input1, input1);
            add_uint64_uint64(input1, out1, out1);
            out1 = multiply_and_reduce_shoup(out1, scale, scale_shoup, obase_value);

            sub_uint64_uint64(obase_value, input2, input2);
            add_uint64_uint64(input2, out2, out2);
            out2 = multiply_and_reduce_shoup(out2, scale, scale_shoup, obase_value);

            st_two_uint64(dst + out_prime_idx * n + degree_idx, out1, out2);
        }
    }

    void DRNSTool::fastbconv_sk(uint64_t *input_base_Bsk, uint64_t *out_base_q, const cudaStream_t &stream) const {
        uint64_t gridDimGlb;

        size_t size_B = base_B_.size();
        size_t size_Bsk = base_Bsk_.size();
        size_t size_Q = base_Q_.size();
        auto n = n_;

        uint64_t *input_base_m_sk = input_base_Bsk + size_B * n;

        auto temp_bconv = make_cuda_auto_ptr<uint64_t>(size_B * n, stream);

        auto temp_m_sk = make_cuda_auto_ptr<uint64_t>(n, stream);

        constexpr int unroll_factor = 2;

        gridDimGlb = size_B * n / unroll_factor / blockDimGlb.x;
        bconv_mult_unroll2_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                temp_bconv.get(), input_base_Bsk, base_B_.QHatInvModq(),
                base_B_.QHatInvModq_shoup(), base_B_.base(), size_B, n);

        gridDimGlb = 1 * n / unroll_factor / blockDimGlb.x;
        bconv_fuse_sub_mul_single_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * 1 * size_B, stream>>>(
                temp_m_sk.get(), temp_bconv.get(), input_base_m_sk, inv_prod_B_mod_m_sk_, inv_prod_B_mod_m_sk_shoup_,
                base_B_to_m_sk_conv_.QHatModp(), base_B_.base(), size_B, m_sk_, 1, n);

        gridDimGlb = size_Q * n / unroll_factor / blockDimGlb.x;
        bconv_matmul_unroll2_kernel<<<
        gridDimGlb, blockDimGlb, sizeof(uint64_t) * size_Q * size_B, stream>>>(
                out_base_q, temp_bconv.get(), base_B_to_q_conv_.QHatModp(), base_B_.base(), size_B, base_Q_.base(),
                size_Q, n);

        gridDimGlb = n_ * base_Ql_.size() / blockDimGlb.x;
        multiply_and_negated_add_rns_poly<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                temp_m_sk.get(), m_sk_.value(),
                prod_B_mod_q(),
                out_base_q, base_Ql_.base(),
                out_base_q, n_,
                base_Ql_.size());
    }

    __global__ void hps_decrypt_scale_and_round_kernel_small(uint64_t *dst, const uint64_t *src,
                                                             const uint64_t *t_QHatInv_mod_q_div_q_mod_t,
                                                             const uint64_t *t_QHatInv_mod_q_div_q_mod_t_shoup,
                                                             const double *t_QHatInv_mod_q_div_q_frac, uint64_t t,
                                                             size_t n, size_t size_Ql) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            double floatSum = 0.0;
            uint64_t intSum = 0;
            uint64_t tmp;
            double tInv = 1. / static_cast<double>(t);

            for (size_t i = 0; i < size_Ql; i++) {
                tmp = src[i * n + tid];
                floatSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
                intSum += multiply_and_reduce_shoup(tmp, t_QHatInv_mod_q_div_q_mod_t[i],
                                                    t_QHatInv_mod_q_div_q_mod_t_shoup[i], t);
            }

            floatSum += static_cast<double>(intSum);
            auto quot = static_cast<uint64_t>(floatSum * tInv);
            floatSum -= static_cast<double>(t * quot);
             
            dst[tid] = llround(floatSum);
        }
    }

    __global__ void hps_decrypt_scale_and_round_kernel_small_lazy(uint64_t *dst, const uint64_t *src,
                                                                  const uint64_t *t_QHatInv_mod_q_div_q_mod_t,
                                                                  const double *t_QHatInv_mod_q_div_q_frac, uint64_t t,
                                                                  size_t n, size_t size_Ql) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            double floatSum = 0.0;
            uint64_t intSum = 0;
            uint64_t tmp;
            double tInv = 1. / static_cast<double>(t);

            for (size_t i = 0; i < size_Ql; i++) {
                tmp = src[i * n + tid];
                floatSum += static_cast<double>(tmp) * t_QHatInv_mod_q_div_q_frac[i];
                intSum += tmp * t_QHatInv_mod_q_div_q_mod_t[i];
            }

            floatSum += static_cast<double>(intSum);
            auto quot = static_cast<uint64_t>(floatSum * tInv);
            floatSum -= static_cast<double>(t * quot);
             
            dst[tid] = llround(floatSum);
        }
    }

    __global__ void hps_decrypt_scale_and_round_kernel_large(
            uint64_t *dst, const uint64_t *src, const uint64_t *t_QHatInv_mod_q_div_q_mod_t,
            const uint64_t *t_QHatInv_mod_q_div_q_mod_t_shoup, const double *t_QHatInv_mod_q_div_q_frac,
            const uint64_t *t_QHatInv_mod_q_B_div_q_mod_t, const uint64_t *t_QHatInv_mod_q_B_div_q_mod_t_shoup,
            const double *t_QHatInv_mod_q_B_div_q_frac, uint64_t t, size_t n, size_t size_Ql, size_t qMSBHf) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            double floatSum = 0.0;
            uint64_t intSum = 0;
            uint64_t tmpLo, tmpHi;
            double tInv = 1. / static_cast<double>(t);

            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t tmp = src[i * n + tid];
                tmpHi = tmp >> qMSBHf;
                tmpLo = tmp & ((1ULL << qMSBHf) - 1);
                floatSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
                floatSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
                intSum += multiply_and_reduce_shoup(tmpLo, t_QHatInv_mod_q_div_q_mod_t[i],
                                                    t_QHatInv_mod_q_div_q_mod_t_shoup[i], t);
                intSum += multiply_and_reduce_shoup(tmpHi, t_QHatInv_mod_q_B_div_q_mod_t[i],
                                                    t_QHatInv_mod_q_B_div_q_mod_t_shoup[i], t);
            }

            floatSum += static_cast<double>(intSum);
            auto quot = static_cast<uint64_t>(floatSum * tInv);
            floatSum -= static_cast<double>(t * quot);
             
            dst[tid] = llround(floatSum);
        }
    }

    __global__ void hps_decrypt_scale_and_round_kernel_large_lazy(uint64_t *dst, const uint64_t *src,
                                                                  const uint64_t *t_QHatInv_mod_q_div_q_mod_t,
                                                                  const double *t_QHatInv_mod_q_div_q_frac,
                                                                  const uint64_t *t_QHatInv_mod_q_B_div_q_mod_t,
                                                                  const double *t_QHatInv_mod_q_B_div_q_frac,
                                                                  uint64_t t, size_t n, size_t size_Ql, size_t qMSBHf) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            double floatSum = 0.0;
            uint64_t intSum = 0;
            uint64_t tmpLo, tmpHi;
            double tInv = 1. / static_cast<double>(t);

            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t tmp = src[i * n + tid];
                tmpHi = tmp >> qMSBHf;
                tmpLo = tmp & ((1ULL << qMSBHf) - 1);
                floatSum += static_cast<double>(tmpLo) * t_QHatInv_mod_q_div_q_frac[i];
                floatSum += static_cast<double>(tmpHi) * t_QHatInv_mod_q_B_div_q_frac[i];
                intSum += tmpLo * t_QHatInv_mod_q_div_q_mod_t[i];
                intSum += tmpHi * t_QHatInv_mod_q_B_div_q_mod_t[i];
            }

            floatSum += static_cast<double>(intSum);
            auto quot = static_cast<uint64_t>(floatSum * tInv);
            floatSum -= static_cast<double>(t * quot);
             
            dst[tid] = llround(floatSum);
        }
    }

    void DRNSTool::hps_decrypt_scale_and_round(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        uint64_t gridDimGlb = n_ / blockDimGlb.x;
        uint64_t t = t_.value();
        size_t n = n_;
        size_t size_Ql = base_Ql_.size();

        if (qMSB_ + sizeQMSB_ < 52) {

            if ((qMSB_ + tMSB_ + sizeQMSB_) < 52) {

                hps_decrypt_scale_and_round_kernel_small_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        dst, src, t_QHatInv_mod_q_div_q_mod_t_.get(), t_QHatInv_mod_q_div_q_frac_.get(), t, n, size_Ql);
            } else {

                hps_decrypt_scale_and_round_kernel_small<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        dst, src, t_QHatInv_mod_q_div_q_mod_t_.get(), t_QHatInv_mod_q_div_q_mod_t_shoup_.get(),
                        t_QHatInv_mod_q_div_q_frac_.get(), t, n, size_Ql);
            }
        } else {
             
            size_t qMSBHf = qMSB_ >> 1;
            if ((qMSBHf + tMSB_ + sizeQMSB_) < 52) {

                hps_decrypt_scale_and_round_kernel_large_lazy<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        dst, src, t_QHatInv_mod_q_div_q_mod_t_.get(), t_QHatInv_mod_q_div_q_frac_.get(),
                        t_QHatInv_mod_q_B_div_q_mod_t_.get(), t_QHatInv_mod_q_B_div_q_frac_.get(), t, n, size_Ql,
                        qMSBHf);
            } else {
                hps_decrypt_scale_and_round_kernel_large<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                        dst, src, t_QHatInv_mod_q_div_q_mod_t_.get(), t_QHatInv_mod_q_div_q_mod_t_shoup_.get(),
                        t_QHatInv_mod_q_div_q_frac_.get(), t_QHatInv_mod_q_B_div_q_mod_t_.get(),
                        t_QHatInv_mod_q_B_div_q_mod_t_shoup_.get(), t_QHatInv_mod_q_B_div_q_frac_.get(), t, n, size_Ql,
                        qMSBHf);
            }
        }
    }

    __device__ inline bool is64BitOverflow(double d) {
         
        constexpr double epsilon = 0.000001;
         
        constexpr int64_t safe_double = 9223372036854775295;
        return ((std::abs(d) - static_cast<double>(safe_double)) > epsilon);
    }

    __global__ void scaleAndRound_HPS_QR_R_kernel(uint64_t *dst, const uint64_t *src,
                                                  const uint64_t *t_R_SHatInv_mod_s_div_s_mod_r,
                                                  const double *t_R_SHatInv_mod_s_div_s_frac, const DModulus *base_Rl,
                                                  size_t n, size_t size_Ql, size_t size_Rl) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            auto src_Ql = src;
            auto src_Rl = src + size_Ql * n;

            double nu = 0.5;
            for (size_t i = 0; i < size_Ql; i++) {
                uint64_t xi = src_Ql[i * n + tid];
                nu += static_cast<double>(xi) * t_R_SHatInv_mod_s_div_s_frac[i];
            }

            auto alpha = static_cast<uint64_t>(nu);

            for (size_t j = 0; j < size_Rl; j++) {
                uint128_t curValue = {0, 0};
                auto rj = base_Rl[j].value();
                auto rj_ratio = base_Rl[j].const_ratio();
                auto t_R_SHatInv_mod_s_div_s_mod_rj = t_R_SHatInv_mod_s_div_s_mod_r + j * (size_Ql + 1);

                for (size_t i = 0; i < size_Ql; i++) {
                    uint64_t xi = src_Ql[i * n + tid];
                    uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[i]);
                    add_uint128_uint128(temp, curValue, curValue);
                }

                uint64_t xi = src_Rl[j * n + tid];
                uint128_t temp = multiply_uint64_uint64(xi, t_R_SHatInv_mod_s_div_s_mod_rj[size_Ql]);
                add_uint128_uint128(temp, curValue, curValue);

                uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, rj, rj_ratio);
                alpha = barrett_reduce_uint64_uint64(alpha, rj, rj_ratio[1]);
                dst[j * n + tid] = add_uint64_uint64_mod(curNativeValue, alpha, rj);
            }
        }
    }

    void DRNSTool::scaleAndRound_HPS_QR_R(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        uint64_t gridDimGlb = n_ / blockDimGlb.x;
        size_t n = n_;
        size_t size_Ql = base_Ql_.size();
        size_t size_Rl = base_Rl_.size();
        scaleAndRound_HPS_QR_R_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, tRSHatInvModsDivsModr(), tRSHatInvModsDivsFrac(), base_Rl_.base(), n, size_Ql, size_Rl);
    }

    __global__ void scaleAndRound_HPS_QlRl_Ql_kernel(uint64_t *dst, const uint64_t *src,
                                                     const uint64_t *tQlSlHatInvModsDivsModq,
                                                     const double *tQlSlHatInvModsDivsFrac, const DModulus *base_Ql,
                                                     size_t n, size_t size_Ql, size_t size_Rl) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
            auto src_Ql = src;
            auto src_Rl = src + size_Ql * n;

            double nu = 0.5;
            for (size_t j = 0; j < size_Rl; j++) {
                uint64_t xj = src_Rl[j * n + tid];
                nu += static_cast<double>(xj) * tQlSlHatInvModsDivsFrac[j];
            }

            auto alpha = static_cast<uint64_t>(nu);

            for (size_t i = 0; i < size_Ql; i++) {
                uint128_t curValue = {0, 0};

                auto tQlSlHatInvModsDivsModqi = tQlSlHatInvModsDivsModq + i * (size_Rl + 1);

                for (size_t j = 0; j < size_Rl; j++) {
                    uint64_t xj = src_Rl[j * n + tid];
                    uint128_t temp = multiply_uint64_uint64(xj, tQlSlHatInvModsDivsModqi[j]);
                    add_uint128_uint128(temp, curValue, curValue);
                }

                uint64_t xi = src_Ql[i * n + tid];
                uint128_t temp = multiply_uint64_uint64(xi, tQlSlHatInvModsDivsModqi[size_Rl]);
                add_uint128_uint128(temp, curValue, curValue);

                auto qi = base_Ql[i].value();
                auto qi_ratio = base_Ql[i].const_ratio();
                uint64_t curNativeValue = barrett_reduce_uint128_uint64(curValue, qi, qi_ratio);
                alpha = barrett_reduce_uint64_uint64(alpha, qi, qi_ratio[1]);
                dst[i * n + tid] = add_uint64_uint64_mod(curNativeValue, alpha, qi);
            }
        }
    }

    void DRNSTool::scaleAndRound_HPS_QlRl_Ql(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        uint64_t gridDimGlb = n_ / blockDimGlb.x;
        size_t n = n_;
        size_t size_Ql = base_Ql_.size();
        size_t size_Rl = base_Rl_.size();
        scaleAndRound_HPS_QlRl_Ql_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, tQlSlHatInvModsDivsModq(), tQlSlHatInvModsDivsFrac(), base_Ql_.base(), n, size_Ql, size_Rl);
    }

    void DRNSTool::scaleAndRound_HPS_Q_Ql(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        uint64_t gridDimGlb = n_ / blockDimGlb.x;
        size_t n = n_;
        size_t size_Ql = base_Ql_.size();
        size_t size_QlDrop = base_QlDrop_.size();
        scaleAndRound_HPS_QlRl_Ql_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, QlQHatInvModqDivqModq(), QlQHatInvModqDivqFrac(), base_Ql_.base(), n, size_Ql, size_QlDrop);
    }

    __global__ void ExpandCRTBasisQlHat_kernel(uint64_t *out, const uint64_t *in, const uint64_t *QlDropModq,
                                               const uint64_t *QlDropModq_shoup, const DModulus *base_Ql,
                                               size_t size_Ql, size_t size_Q, uint64_t n) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Q; tid += blockDim.x * gridDim.x) {
            size_t i = tid / n;
            if (i < size_Ql) {
                auto modulus = base_Ql[i].value();
                out[tid] = multiply_and_reduce_shoup(in[tid], QlDropModq[i], QlDropModq_shoup[i], modulus);
            } else {
                out[tid] = 0;
            }
        }
    }

    void DRNSTool::ExpandCRTBasis_Ql_Q(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        size_t size_Ql = base_Ql_.size();
        size_t size_Q = base_Q_.size();

        size_t n = n_;
        uint64_t gridDimGlb = n * size_Q / blockDimGlb.x;
        ExpandCRTBasisQlHat_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, base_Ql_to_QlDrop_conv_.PModq(),
                base_Ql_to_QlDrop_conv_.PModq_shoup(), base_Ql_.base(),
                size_Ql, size_Q, n);
    }

    __global__ void ExpandCRTBasisQlHat_add_to_ct_kernel(uint64_t *out, const uint64_t *in, const uint64_t *QlDropModq,
                                                         const uint64_t *QlDropModq_shoup, const DModulus *base_Ql,
                                                         size_t size_Ql, uint64_t n) {
        for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n * size_Ql; tid += blockDim.x * gridDim.x) {
            size_t i = tid / n;
            auto modulus = base_Ql[i].value();
            uint64_t tmp = multiply_and_reduce_shoup(in[tid], QlDropModq[i], QlDropModq_shoup[i], modulus);
            out[tid] = add_uint64_uint64_mod(tmp, out[tid], modulus);
        }
    }

    void DRNSTool::ExpandCRTBasis_Ql_Q_add_to_ct(uint64_t *dst, const uint64_t *src, const cudaStream_t &stream) const {
        size_t size_Ql = base_Ql_.size();

        size_t n = n_;
        uint64_t gridDimGlb = n * size_Ql / blockDimGlb.x;
        ExpandCRTBasisQlHat_add_to_ct_kernel<<<gridDimGlb, blockDimGlb, 0, stream>>>(
                dst, src, base_Ql_to_QlDrop_conv_.PModq(), base_Ql_to_QlDrop_conv_.PModq_shoup(),
                base_Ql_.base(), size_Ql, n);
    }
}
