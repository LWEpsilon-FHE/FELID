#include "RbootSlimPolyEval.cuh"

#include <cmath>
#include <stdexcept>

using namespace NTL;
using namespace phantom;
using namespace moai;
using namespace phantom::util;

namespace rboot_slim {
namespace {

long count_ones(long value)
{
    return static_cast<long>(__builtin_popcountl(static_cast<unsigned long>(value)));
}

void gene_t0_t1(const RbootSparseOps &ops,
                PhantomCiphertext &t0,
                PhantomCiphertext &t1,
                const PhantomCiphertext &cipher,
                const cuda_stream_wrapper &stream_wrapper)
{
    const double scale = cipher.scale();
    std::vector<double> ones(ops.full_slot_count(), 1.0);
    PhantomPlaintext plain_one;
    ops.encode_exact_vector(ones, scale, plain_one, 1, stream_wrapper);

    PhantomCiphertext ctxt_one;
    ops.ckks->encryptor.encrypt(plain_one, ctxt_one, stream_wrapper);

    t0 = ctxt_one;
    t1 = cipher;
}

void eval_t(const RbootSparseOps &ops,
            PhantomCiphertext &tm_plus_n,
            const PhantomCiphertext &tm,
            const PhantomCiphertext &tn,
            const PhantomCiphertext &tm_minus_n,
            const cuda_stream_wrapper &stream_wrapper)
{
    PhantomCiphertext temp;
    ops.multiply_reduced_error(tm, tn, *(ops.ckks->relin_keys), temp, stream_wrapper);
    ops.ckks->evaluator.add_inplace(temp, temp, stream_wrapper);
    ops.ckks->evaluator.rescale_to_next_inplace(temp, stream_wrapper);
    ops.sub_reduced_error(temp, tm_minus_n, tm_plus_n, stream_wrapper);
}

}   

long pow2(long value)
{
    return 1L << value;
}

long ceil_to_int(double value)
{
    return static_cast<long>(std::ceil(value));
}

void upgrade_oddbaby(long n, minicomp::Tree &tree)
{
    const long depth = ceil_to_int(std::log(static_cast<double>(n)) / std::log(2.0));
    long best_total = 10000;
    long best_m = 0;
    long best_l = 0;
    minicomp::Tree best_tree;
    minicomp::Tree candidate_tree;

    for (long l = 1; pow2(l) - 1 <= n; ++l)
    {
        for (long m = 1; pow2(m - 1) < n; ++m)
        {
            std::vector<std::vector<int>> cost(n + 1, std::vector<int>(depth + 1, 0));
            std::vector<std::vector<minicomp::Tree>> trees(
                n + 1, std::vector<minicomp::Tree>(depth + 1, minicomp::Tree(evaltype::oddbaby)));

            cost[1][1] = 0;
            for (int i = 3; i <= n; i += 2)
            {
                cost[i][1] = 10000;
            }

            for (int stage = 2; stage <= depth; ++stage)
            {
                for (int degree = 1; degree <= n; degree += 2)
                {
                    if (degree <= pow2(l) - 1 && degree <= pow2(stage - 1))
                    {
                        cost[degree][stage] = 0;
                        continue;
                    }

                    int best = 10000;
                    candidate_tree.clear();
                    for (int k = 1; k <= m - 1 && pow2(k) < degree && k < stage; ++k)
                    {
                        const long g = pow2(k);
                        const int current = cost[degree - g][stage - 1] + cost[g - 1][stage] + 1;
                        if (current < best)
                        {
                            best = current;
                            candidate_tree.merge(trees[g - 1][stage], trees[degree - g][stage - 1], static_cast<int>(g));
                        }
                    }
                    cost[degree][stage] = best;
                    trees[degree][stage] = candidate_tree;
                }
            }

            const int total_cost = cost[n][depth] + static_cast<int>(pow2(l - 1) + m - 2);
            if (total_cost < best_total)
            {
                best_total = total_cost;
                best_tree = trees[n][depth];
                best_m = m;
                best_l = l;
            }
        }
    }

    tree = best_tree;
    tree.m = static_cast<int>(best_m);
    tree.l = static_cast<int>(best_l);
}

void upgrade_baby(long n, minicomp::Tree &tree)
{
    const long depth = ceil_to_int(std::log(static_cast<double>(n + 1)) / std::log(2.0));
    long best_total = 10000;
    long best_m = 0;
    long best_b = 0;
    minicomp::Tree best_tree;
    minicomp::Tree candidate_tree;

    if (n == 1)
    {
        tree = minicomp::Tree(evaltype::baby);
        tree.m = 1;
        tree.b = 1;
        return;
    }

    for (long b = 1; b <= n; ++b)
    {
        for (long m = 1; pow2(m - 1) * b <= n; ++m)
        {
            std::vector<std::vector<int>> cost(n + 1, std::vector<int>(depth + 1, 0));
            std::vector<std::vector<minicomp::Tree>> trees(
                n + 1, std::vector<minicomp::Tree>(depth + 1, minicomp::Tree(evaltype::baby)));

            for (int stage = 1; stage <= depth; ++stage)
            {
                for (int degree = 1; degree <= n; ++degree)
                {
                    if (degree + 1 > pow2(stage))
                    {
                        cost[degree][stage] = 10000;
                        trees[degree][stage] = minicomp::Tree(evaltype::baby);
                        continue;
                    }

                    if (((b == 1 && m >= 2) && degree <= 2 && degree <= pow2(stage - 1)) ||
                        (degree <= b && degree <= pow2(stage - 1)))
                    {
                        cost[degree][stage] = 0;
                        trees[degree][stage] = minicomp::Tree(evaltype::baby);
                        continue;
                    }

                    int best = 10000;
                    candidate_tree.clear();

                    for (int k = 2; k <= b; ++k)
                    {
                        const long g = k;
                        if (g <= pow2(stage - 1) && g < degree && cost[degree - g][stage - 1] + cost[g - 1][stage] + 1 < best)
                        {
                            best = cost[degree - g][stage - 1] + cost[g - 1][stage] + 1;
                            candidate_tree.merge(trees[g - 1][stage], trees[degree - g][stage - 1], static_cast<int>(g));
                        }
                    }

                    for (int k = 0; k <= m - 1; ++k)
                    {
                        const long g = pow2(k) * b;
                        if (g <= pow2(stage - 1) && g >= 2 && g < degree &&
                            cost[degree - g][stage - 1] + cost[g - 1][stage] + 1 < best)
                        {
                            best = cost[degree - g][stage - 1] + cost[g - 1][stage] + 1;
                            candidate_tree.merge(trees[g - 1][stage], trees[degree - g][stage - 1], static_cast<int>(g));
                        }
                    }

                    cost[degree][stage] = best;
                    trees[degree][stage] = candidate_tree;
                }
            }

            const int total_cost = cost[n][depth] + static_cast<int>(m + b - 2);
            if (total_cost < best_total)
            {
                best_total = total_cost;
                best_tree = trees[n][depth];
                best_m = m;
                best_b = b;
            }
        }
    }

    tree = best_tree;
    tree.m = static_cast<int>(best_m);
    tree.b = static_cast<int>(best_b);
}

long coeff_number(long deg, minicomp::Tree &tree)
{
    long total = 0;
    std::vector<long> decomp_deg(pow2(tree.depth + 1), -1);
    decomp_deg[1] = deg;

    for (int level = 1; level <= tree.depth; ++level)
    {
        for (int index = pow2(level); index < pow2(level + 1); ++index)
        {
            if (index % 2 == 0)
            {
                decomp_deg[index] = tree.tree[index / 2] - 1;
            }
            else
            {
                decomp_deg[index] = decomp_deg[index / 2] - tree.tree[index / 2];
            }
        }
    }

    for (int index = 0; index < pow2(tree.depth + 1); ++index)
    {
        if (tree.tree[index] == 0)
        {
            total += decomp_deg[index] + 1;
        }
    }
    return total;
}

void homomorphic_poly_evaluation_sparse(const boot::Polynomial &poly,
                                        const RbootSparseOps &ops,
                                        PhantomCiphertext &rtn,
                                        const PhantomCiphertext &cipher,
                                        const cuda_stream_wrapper &stream_wrapper)
{
    const double zero = 1.0 / cipher.scale();

    if (poly.deg == 1)
    {
        ops.multiply_const(cipher, to_double(poly.coeff[1]), rtn, stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(rtn, stream_wrapper);
        ops.add_const(rtn, to_double(poly.coeff[0]), rtn, stream_wrapper);
        return;
    }

    if (poly.deg == 2)
    {
        PhantomCiphertext squared;
        ops.ckks->evaluator.square(const_cast<PhantomCiphertext &>(cipher), squared, stream_wrapper);
        ops.ckks->evaluator.relinearize_inplace(squared, *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(squared, stream_wrapper);

        ops.multiply_const_inplace(squared, to_double(poly.coeff[2]), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(squared, stream_wrapper);

        if (std::abs(to_double(poly.coeff[1])) >= zero)
        {
            ops.multiply_const(cipher, to_double(poly.coeff[1]), rtn, stream_wrapper);
            ops.ckks->evaluator.rescale_to_next_inplace(rtn, stream_wrapper);
            ops.add_reduced_error(rtn, squared, rtn, stream_wrapper);
        }
        else
        {
            rtn = squared;
        }

        ops.add_const_inplace(rtn, to_double(poly.coeff[0]), stream_wrapper);
        return;
    }

    if (poly.deg == 3)
    {
        PhantomCiphertext squared;
        PhantomCiphertext cubic;
        ops.ckks->evaluator.square(const_cast<PhantomCiphertext &>(cipher), squared, stream_wrapper);
        ops.ckks->evaluator.relinearize_inplace(squared, *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(squared, stream_wrapper);

        ops.multiply_const(cipher, to_double(poly.coeff[3]), cubic, stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(cubic, stream_wrapper);

        ops.multiply_inplace_reduced_error(cubic, squared, *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(cubic, stream_wrapper);

        if (std::abs(to_double(poly.coeff[1])) >= zero)
        {
            ops.multiply_const(cipher, to_double(poly.coeff[1]), rtn, stream_wrapper);
            ops.ckks->evaluator.rescale_to_next_inplace(rtn, stream_wrapper);
            ops.add_reduced_error(rtn, cubic, rtn, stream_wrapper);
        }
        else
        {
            rtn = cubic;
        }

        if (std::abs(to_double(poly.coeff[2])) >= zero)
        {
            ops.multiply_const_inplace(squared, to_double(poly.coeff[2]), stream_wrapper);
            ops.ckks->evaluator.rescale_to_next_inplace(squared, stream_wrapper);
            ops.add_reduced_error(rtn, squared, rtn, stream_wrapper);
        }

        ops.add_const_inplace(rtn, to_double(poly.coeff[0]), stream_wrapper);
        return;
    }

    std::vector<PhantomCiphertext> baby(poly.heap_k, PhantomCiphertext());
    std::vector<bool> babybool(poly.heap_k, false);
    baby[1] = cipher;
    babybool[1] = true;

    for (int i = 2; i < poly.heap_k; i *= 2)
    {
        ops.ckks->evaluator.square(baby[i / 2], baby[i], stream_wrapper);
        ops.ckks->evaluator.relinearize_inplace(baby[i], *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(baby[i], stream_wrapper);
        ops.ckks->evaluator.double_inplace(baby[i], stream_wrapper);
        ops.add_const(baby[i], -1.0, baby[i], stream_wrapper);
        babybool[i] = true;
    }

    long lpow2;
    long res_idx;
    long diff;
    PhantomCiphertext tmp;

    for (int i = 1; i < poly.heap_k; ++i)
    {
        if (!babybool[i])
        {
            lpow2 = (1 << static_cast<int>(std::floor(std::log(static_cast<double>(i)) / std::log(2.0))));
            res_idx = i - lpow2;
            diff = std::llabs(lpow2 - res_idx);

            ops.multiply_reduced_error(baby[lpow2], baby[res_idx], *(ops.ckks->relin_keys), baby[i], stream_wrapper);
            ops.ckks->evaluator.rescale_to_next_inplace(baby[i], stream_wrapper);
            ops.ckks->evaluator.double_inplace(baby[i], stream_wrapper);
            ops.sub_reduced_error(baby[i], baby[diff], baby[i], stream_wrapper);
            babybool[i] = true;
        }
    }

    std::vector<PhantomCiphertext> giant(poly.heap_m, PhantomCiphertext());
    lpow2 = (1 << (static_cast<int>(std::ceil(std::log(static_cast<double>(poly.heap_k)) / std::log(2.0))) - 1));
    res_idx = poly.heap_k - lpow2;
    diff = std::llabs(lpow2 - res_idx);

    if (res_idx == 0)
    {
        giant[0] = baby[lpow2];
    }
    else if (diff == 0)
    {
        ops.ckks->evaluator.square(baby[lpow2], giant[0], stream_wrapper);
        ops.ckks->evaluator.relinearize_inplace(giant[0], *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(giant[0], stream_wrapper);
        ops.ckks->evaluator.double_inplace(giant[0], stream_wrapper);
        ops.add_const(giant[0], -1.0, giant[0], stream_wrapper);
    }
    else
    {
        ops.multiply_reduced_error(baby[lpow2], baby[res_idx], *(ops.ckks->relin_keys), giant[0], stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(giant[0], stream_wrapper);
        ops.ckks->evaluator.double_inplace(giant[0], stream_wrapper);
        ops.sub_reduced_error(giant[0], baby[diff], giant[0], stream_wrapper);
    }

    for (int i = 1; i < poly.heap_m; ++i)
    {
        ops.ckks->evaluator.square(giant[i - 1], giant[i], stream_wrapper);
        ops.ckks->evaluator.relinearize_inplace(giant[i], *(ops.ckks->relin_keys), stream_wrapper);
        ops.ckks->evaluator.rescale_to_next_inplace(giant[i], stream_wrapper);
        ops.ckks->evaluator.double_inplace(giant[i], stream_wrapper);
        ops.add_const_inplace(giant[i], -1.0, stream_wrapper);
    }

    std::vector<PhantomCiphertext> cipherheap((1 << (poly.heap_m + 1)) - 1, PhantomCiphertext());
    std::vector<bool> cipherheapbool((1 << (poly.heap_m + 1)) - 1, false);
    long heapfirst = (1 << poly.heap_m) - 1;
    long heaplast = (1 << (poly.heap_m + 1)) - 1;

    for (int i = heapfirst; i < heaplast; ++i)
    {
        if (poly.poly_heap[i])
        {
            cipherheapbool[i] = true;
            ops.multiply_const(baby[1], poly.poly_heap[i]->deg >= 1 ? to_double(poly.poly_heap[i]->chebcoeff[1]) : 0.0, cipherheap[i], stream_wrapper);
            ops.ckks->evaluator.rescale_to_next_inplace(cipherheap[i], stream_wrapper);

            ops.add_const_inplace(cipherheap[i], to_double(poly.poly_heap[i]->chebcoeff[0]), stream_wrapper);

            for (int j = 2; j <= poly.poly_heap[i]->deg; ++j)
            {
                if (std::abs(to_double(poly.poly_heap[i]->chebcoeff[j])) <= zero)
                {
                    continue;
                }

                if (j < poly.heap_k)
                {
                    ops.multiply_const(baby[j], to_double(poly.poly_heap[i]->chebcoeff[j]), tmp, stream_wrapper);
                }
                else
                {
                    ops.multiply_const(giant[0], to_double(poly.poly_heap[i]->chebcoeff[j]), tmp, stream_wrapper);
                }
                ops.ckks->evaluator.rescale_to_next_inplace(tmp, stream_wrapper);
                ops.add_reduced_error(cipherheap[i], tmp, cipherheap[i], stream_wrapper);
            }
        }
    }

    long depth = poly.heap_m;
    long giant_index = 0;
    while (depth != 0)
    {
        --depth;
        heapfirst = (1 << depth) - 1;
        heaplast = (1 << (depth + 1)) - 1;
        for (int i = heapfirst; i < heaplast; ++i)
        {
            if (poly.poly_heap[i])
            {
                cipherheapbool[i] = true;
                if (!cipherheapbool[2 * (i + 1) - 1])
                {
                    cipherheap[i] = cipherheap[2 * (i + 1)];
                }
                else
                {
                    ops.multiply_reduced_error(cipherheap[2 * (i + 1) - 1], giant[giant_index], *(ops.ckks->relin_keys), cipherheap[i], stream_wrapper);
                    ops.ckks->evaluator.rescale_to_next_inplace(cipherheap[i], stream_wrapper);
                    ops.add_reduced_error(cipherheap[i], cipherheap[2 * (i + 1)], cipherheap[i], stream_wrapper);
                }
            }
        }
        ++giant_index;
    }

    rtn = cipherheap[0];
}

void eval_polynomial_integrate(const RbootSparseOps &ops,
                               PhantomCiphertext &res,
                               const PhantomCiphertext &cipher,
                               long deg,
                               const std::vector<RR> &decomp_coeff,
                               minicomp::Tree &tree,
                               const cuda_stream_wrapper &stream_wrapper)
{
    const double scale = cipher.scale();
    const long total_depth = ceil_to_int(std::log(static_cast<double>(deg + 1)) / std::log(2.0));
    const evaltype eval_type = tree.type;
    if (eval_type != evaltype::oddbaby && eval_type != evaltype::baby)
    {
        throw std::invalid_argument("Rboot slim polynomial integration only supports oddbaby or baby trees.");
    }

    std::vector<long> decomp_deg(pow2(tree.depth + 1), -1);
    std::vector<long> start_index(pow2(tree.depth + 1), -1);
    std::vector<std::unique_ptr<PhantomCiphertext>> T(deg + 10);
    std::vector<std::unique_ptr<PhantomCiphertext>> pt(deg + 10);

    T[0] = std::make_unique<PhantomCiphertext>();
    T[1] = std::make_unique<PhantomCiphertext>();

    long temp_index = (eval_type == evaltype::oddbaby) ? 1 : 0;
    decomp_deg[1] = deg;
    for (int level = 1; level <= tree.depth; ++level)
    {
        for (int index = pow2(level); index < pow2(level + 1); ++index)
        {
            if (index % 2 == 0)
            {
                decomp_deg[index] = tree.tree[index / 2] - 1;
            }
            else
            {
                decomp_deg[index] = decomp_deg[index / 2] - tree.tree[index / 2];
            }
        }
    }

    for (int index = 1; index < pow2(tree.depth + 1); ++index)
    {
        if (tree.tree[index] == 0)
        {
            start_index[index] = temp_index;
            temp_index += decomp_deg[index] + 1;
        }
    }

    gene_t0_t1(ops, *T[0], *T[1], cipher, stream_wrapper);

    PhantomPlaintext zero_plain;
    std::vector<double> zeros(ops.full_slot_count(), 0.0);
    ops.encode_exact_vector(zeros, scale * scale, zero_plain, 1, stream_wrapper);
    PhantomCiphertext zero_cipher;
    ops.ckks->encryptor.encrypt(zero_plain, zero_cipher, stream_wrapper);

    PhantomCiphertext temp1;
    for (int stage = 1; stage <= total_depth; ++stage)
    {
        for (int index = 1; index < pow2(tree.depth + 1); ++index)
        {
            if (tree.tree[index] != 0 || total_depth + 1 - count_ones(index) != stage)
            {
                continue;
            }

            long coeff_index = start_index[index];
            pt[index] = std::make_unique<PhantomCiphertext>();
            if (eval_type == evaltype::oddbaby)
            {
                ops.multiply_const(*T[1], to_double(decomp_coeff.at(static_cast<size_t>(coeff_index))), *pt[index], stream_wrapper);
                coeff_index += 2;

                for (int term = 3; term <= decomp_deg[index]; term += 2)
                {
                    if (!T[term])
                    {
                        throw std::runtime_error("Chebyshev basis ciphertext missing during oddbaby evaluation.");
                    }
                    ops.multiply_const(*T[term], to_double(decomp_coeff.at(static_cast<size_t>(coeff_index))), temp1, stream_wrapper);
                    ops.add_inplace_reduced_error(*pt[index], temp1, stream_wrapper);
                    coeff_index += 2;
                }
            }
            else
            {
                *pt[index] = zero_cipher;
                for (int term = 0; term <= decomp_deg[index]; ++term, ++coeff_index)
                {
                    if (std::abs(to_double(decomp_coeff.at(static_cast<size_t>(coeff_index)))) <= 1.0 / scale)
                    {
                        continue;
                    }
                    if (!T[term])
                    {
                        throw std::runtime_error("Chebyshev basis ciphertext missing during baby evaluation.");
                    }
                    ops.multiply_const(*T[term], to_double(decomp_coeff.at(static_cast<size_t>(coeff_index))), temp1, stream_wrapper);
                    ops.add_inplace_reduced_error(*pt[index], temp1, stream_wrapper);
                }
            }

            ops.ckks->evaluator.rescale_to_next_inplace(*pt[index], stream_wrapper);
        }

        if (eval_type == evaltype::oddbaby)
        {
            for (int index = 1; index < pow2(tree.depth + 1); ++index)
            {
                if (tree.tree[index] <= 0 || total_depth + 1 - count_ones(index) != stage || index % 2 == 0)
                {
                    continue;
                }

                long node = index;
                pt[index] = std::make_unique<PhantomCiphertext>();
                if (!T[tree.tree[node]] || !pt[2 * node + 1])
                {
                    throw std::runtime_error("Intermediate oddbaby node is missing.");
                }

                ops.multiply_reduced_error(*T[tree.tree[node]], *pt[2 * node + 1], *(ops.ckks->relin_keys), *pt[index], stream_wrapper);
                node *= 2;
                while (true)
                {
                    if (tree.tree[node] == 0)
                    {
                        break;
                    }
                    if (!T[tree.tree[node]] || !pt[2 * node + 1])
                    {
                        throw std::runtime_error("Intermediate oddbaby branch is missing.");
                    }
                    ops.multiply_reduced_error(*T[tree.tree[node]], *pt[2 * node + 1], *(ops.ckks->relin_keys), temp1, stream_wrapper);
                    ops.add_inplace_reduced_error(*pt[index], temp1, stream_wrapper);
                    node *= 2;
                }
                ops.ckks->evaluator.rescale_to_next_inplace(*pt[index], stream_wrapper);
                ops.add_inplace_reduced_error(*pt[index], *pt[node], stream_wrapper);
            }

            if (stage <= tree.m - 1)
            {
                const int cheb_index = static_cast<int>(pow2(stage));
                T[cheb_index] = std::make_unique<PhantomCiphertext>();
                eval_t(ops, *T[cheb_index], *T[cheb_index / 2], *T[cheb_index / 2], *T[0], stream_wrapper);
            }

            if (stage <= tree.l)
            {
                const int base = static_cast<int>(pow2(stage - 1));
                const int limit = static_cast<int>(pow2(stage));
                for (int index = base + 1; index <= limit - 1; index += 2)
                {
                    T[index] = std::make_unique<PhantomCiphertext>();
                    eval_t(ops, *T[index], *T[base], *T[index - base], *T[limit - index], stream_wrapper);
                }
            }
            continue;
        }

        std::vector<long> inter;
        for (int index = 1; index < pow2(tree.depth + 1); ++index)
        {
            if (tree.tree[index] <= 0 || total_depth + 1 - count_ones(index) != stage)
            {
                continue;
            }

            bool no_execute = false;
            for (long seen : inter)
            {
                long node = index;
                while (true)
                {
                    if (node == seen)
                    {
                        no_execute = true;
                        break;
                    }
                    if (node % 2 == 0)
                    {
                        node /= 2;
                    }
                    else
                    {
                        break;
                    }
                }
                if (no_execute)
                {
                    break;
                }
            }
            if (no_execute)
            {
                continue;
            }

            inter.push_back(index);
            long node = index;
            pt[index] = std::make_unique<PhantomCiphertext>();
            if (!T[tree.tree[node]] || !pt[2 * node + 1])
            {
                throw std::runtime_error("Intermediate baby node is missing.");
            }

            ops.multiply_reduced_error(*T[tree.tree[node]], *pt[2 * node + 1], *(ops.ckks->relin_keys), *pt[index], stream_wrapper);
            node *= 2;
            while (true)
            {
                if (tree.tree[node] == 0)
                {
                    break;
                }
                if (!T[tree.tree[node]] || !pt[2 * node + 1])
                {
                    throw std::runtime_error("Intermediate baby branch is missing.");
                }
                ops.multiply_reduced_error(*T[tree.tree[node]], *pt[2 * node + 1], *(ops.ckks->relin_keys), temp1, stream_wrapper);
                ops.add_inplace_reduced_error(*pt[index], temp1, stream_wrapper);
                node *= 2;
            }
            ops.ckks->evaluator.rescale_to_next_inplace(*pt[index], stream_wrapper);
            ops.add_inplace_reduced_error(*pt[index], *pt[node], stream_wrapper);
        }

        for (int g = 2; g <= tree.b; ++g)
        {
            if (!(pow2(stage - 1) < g && g <= pow2(stage)))
            {
                continue;
            }

            T[g] = std::make_unique<PhantomCiphertext>();
            if (g % 2 == 0)
            {
                eval_t(ops, *T[g], *T[g / 2], *T[g / 2], *T[0], stream_wrapper);
            }
            else
            {
                eval_t(ops, *T[g], *T[g / 2], *T[(g + 1) / 2], *T[1], stream_wrapper);
            }
        }

        for (int j = 1; j <= tree.m - 1; ++j)
        {
            const int g = static_cast<int>(pow2(j) * tree.b);
            if (!(pow2(stage - 1) < g && g <= pow2(stage)))
            {
                continue;
            }

            T[g] = std::make_unique<PhantomCiphertext>();
            if (g % 2 == 0)
            {
                eval_t(ops, *T[g], *T[g / 2], *T[g / 2], *T[0], stream_wrapper);
            }
            else
            {
                eval_t(ops, *T[g], *T[g / 2], *T[(g + 1) / 2], *T[1], stream_wrapper);
            }
        }
    }

    if (!pt[1])
    {
        throw std::runtime_error("Oddbaby polynomial evaluation did not produce the root node.");
    }
    res = *pt[1];
}

}
