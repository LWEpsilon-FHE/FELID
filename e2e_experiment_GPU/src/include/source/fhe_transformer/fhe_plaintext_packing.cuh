#pragma once

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/fhe_transformer/phantom_adapter.cuh"

namespace fhe_transformer {

template <typename SlotT>
inline void encode_encrypt_at_logical_top(const std::vector<SlotT> &values,
                                          double scale,
                                          const PhantomContext &context,
                                          const FHELevelMap &level_map,
                                          moai::Encoder &encoder,
                                          moai::Encryptor &encryptor,
                                          PhantomCiphertext &ct)
{
    PhantomPlaintext plain;
    encoder.encode(values, context.get_first_index(), scale, plain);
    encryptor.encrypt(plain, ct);
    if (ct.chain_index() < level_map.top_chain_index)
    {
        ::mod_switch_to_inplace(context, ct, level_map.top_chain_index);
    }
}

inline std::vector<double> vec_left_rot(const std::vector<double> &v, int r)
{
    const int size = static_cast<int>(v.size());
    std::vector<double> output(size, 0.0);
    for (int i = 0; i < size; ++i)
    {
        output[i] = v[((i + r) % size + size) % size];
    }
    return output;
}

inline std::vector<std::complex<double>> vec_left_rot_complex(const std::vector<std::complex<double>> &v, int r)
{
    const int size = static_cast<int>(v.size());
    std::vector<std::complex<double>> output(size, 0.0);
    for (int i = 0; i < size; ++i)
    {
        output[i] = v[((i + r) % size + size) % size];
    }
    return output;
}

inline FHEBlockTensor modified_col_packing_batch_encrypt(
    const std::vector<std::vector<std::vector<double>>> &x,
    int batch_size,
    int num_row,
    int num_col,
    int k,
    double scale,
    const PhantomContext &context,
    const FHELevelMap &level_map,
    moai::Encoder &encoder,
    moai::Encryptor &encryptor,
    const std::string &name)
{
    const int n = static_cast<int>(encoder.slot_count());
    const int n2 = n / batch_size;
    const int num_ct = num_col * num_row * batch_size / n;
    if (num_ct <= 0 || num_col * num_row * batch_size % n != 0)
    {
        throw std::runtime_error("modified_col_packing_batch_encrypt dimensions do not produce integral ciphertext count");
    }

    FHEBlockTensor output;
    output.name = name;
    output.logical_shape = {num_row, num_col};
    output.packing = PackingType::InterleavedColumn;
    output.is_complex = false;
    output.cts.resize(num_ct);

    for (int i = 0; i < num_ct; ++i)
    {
        std::vector<double> vec(n, 0.0);
        for (int j = 0; j < n2; ++j)
        {
            for (int ind = 0; ind < batch_size; ++ind)
            {
                const int row_index = j % num_row;
                const int col_index = k * ((j / num_row) % (n2 / (k * num_row))) +
                                      (k * j) / n2 +
                                      i * (n2 / num_row);
                vec[j * batch_size + ind] = x[ind][row_index][col_index];
            }
        }
        encode_encrypt_at_logical_top(vec, scale, context, level_map, encoder, encryptor, output.cts[i]);
    }

    refresh_tensor_metadata(context, level_map, output);
    return output;
}

inline std::vector<std::vector<std::complex<double>>> modified_col_packing_batch_complex_pt(
    const std::vector<std::vector<std::vector<double>>> &x,
    int batch_size,
    int num_row,
    int num_col,
    int k,
    int slot_count)
{
    const int n = slot_count;
    const int n2 = n / batch_size;
    const int num_ct = num_col * num_row * batch_size / (2 * n);
    if (num_ct <= 0 || num_col * num_row * batch_size % (2 * n) != 0)
    {
        throw std::runtime_error("modified_col_packing_batch_complex_pt dimensions do not produce integral ciphertext count");
    }

    std::vector<std::vector<std::complex<double>>> output(num_ct, std::vector<std::complex<double>>(n, 0.0));
    for (int i = 0; i < num_ct; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            for (int ind = 0; ind < batch_size; ++ind)
            {
                const int row_index = j % num_row;
                const int col_index_real = k * ((j / num_row) % (n2 / (k * num_row))) +
                                           (k * j) / n2 +
                                           (2 * i) * (n2 / num_row);
                const int col_index_imag = k * ((j / num_row) % (n2 / (k * num_row))) +
                                           (k * j) / n2 +
                                           (2 * i + 1) * (n2 / num_row);
                output[i][j * batch_size + ind] =
                    std::complex<double>(x[ind][row_index][col_index_real],
                                         x[ind][row_index][col_index_imag]);
            }
        }
    }
    return output;
}

inline std::vector<std::vector<double>> modified_col_packing_batch_pt(
    const std::vector<std::vector<double>> &x,
    int batch_size,
    int num_row,
    int num_col,
    int k,
    int slot_count)
{
    const int n = slot_count;
    const int n2 = n / batch_size;
    const int num_ct = num_col * num_row * batch_size / n;
    if (num_ct <= 0 || num_col * num_row * batch_size % n != 0)
    {
        throw std::runtime_error("modified_col_packing_batch_pt dimensions do not produce integral ciphertext count");
    }

    std::vector<std::vector<double>> output(num_ct, std::vector<double>(n, 0.0));
    for (int i = 0; i < num_ct; ++i)
    {
        for (int j = 0; j < n2; ++j)
        {
            for (int ind = 0; ind < batch_size; ++ind)
            {
                const int row_index = j % num_row;
                const int col_index = k * ((j / num_row) % (n2 / (k * num_row))) +
                                      (k * j) / n2 +
                                      i * (n2 / num_row);
                output[i][j * batch_size + ind] = x[row_index][col_index];
            }
        }
    }
    return output;
}

inline FHEBlockTensor lora_a_packing_batch_encrypt(
    const std::vector<std::vector<double>> &w,
    int batch_size,
    int row_x,
    int num_row,
    int num_col,
    double scale,
    const PhantomContext &context,
    const FHELevelMap &level_map,
    moai::Encoder &encoder,
    moai::Encryptor &encryptor,
    const std::string &name)
{
    const int n = static_cast<int>(encoder.slot_count());
    const int num_ct_x = row_x * num_row * batch_size / n;
    const int num_ct = num_col * num_ct_x;
    if (num_ct_x <= 0 || row_x * num_row * batch_size % n != 0)
    {
        throw std::runtime_error("lora_a_packing_batch_encrypt dimensions do not produce integral ciphertext count");
    }

    FHEBlockTensor output;
    output.name = name;
    output.logical_shape = {num_row, num_col};
    output.packing = PackingType::RepeatedColumn;
    output.is_complex = false;
    output.cts.resize(num_ct);

    for (int i = 0; i < num_col; ++i)
    {
        const int n2 = num_row / num_ct_x;
        for (int j = 0; j < num_ct_x; ++j)
        {
            std::vector<double> vec(n, 0.0);
            for (int ind = 0; ind < n2; ++ind)
            {
                for (int ind2 = 0; ind2 < row_x * batch_size; ++ind2)
                {
                    vec[ind * row_x * batch_size + ind2] = w[j * n2 + ind][i];
                }
            }
            encode_encrypt_at_logical_top(vec, scale, context, level_map, encoder, encryptor,
                                          output.cts[i * num_ct_x + j]);
        }
    }

    refresh_tensor_metadata(context, level_map, output);
    return output;
}

}
