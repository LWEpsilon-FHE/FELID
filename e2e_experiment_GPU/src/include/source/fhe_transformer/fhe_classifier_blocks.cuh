#pragma once

#include <string>

#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/fhe_classifier_ops.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

struct ClassifierForwardConfig {
    PhantomBootstrappingCallbacks callbacks;
    int x1_extract_input_level = 6;
    int x1_input_level = 5;
    int x1w_c1_level = 3;
    int tanh_classifier_level = 9;
    int x1w_c2_level = 8;
    int bts2_stored_level = 3;
    MatrixOpParams c1_params{1, 768, 128, 768, 768, 1};
    MatrixOpParams c2_params{1, 768, 1, 2, 768, 1};

    std::function<FHEBlockTensor(FHEAdapterContext &, const FHEBlockTensor &, int,
                                 const std::string &)> probability_backend;
    std::vector<double> bias_c1, bias_c2;
};

struct ClassifierForwardResult {
    FHEBlockTensor X1Wc1;
    FHEBlockTensor tanh_classifier;
    FHEBlockTensor X1Wc2;
    FHEBlockTensor y_pred;
    FHEBlockTensor y_lr;
};

struct ClassifierBackwardConfig {
    PhantomBootstrappingCallbacks callbacks;
    int grad_Wc2_input_level = 3;
    int grad_Wc2_level = 2;
    int pre_input_level = 3;
    int pre_level = 0;
    int bts3_stored_level = 10;
    int grad_Wc1_x_input_level = 11;
    int grad_Wc1_level = 9;
    int grad_x2_input_level = 9;
    int grad_x2_level = 7;
    ClassifierBackwardOpParams cpu_params;

    bool write_cache = true;
};

struct ClassifierBackwardResult {
    FHEBlockTensor grad_Wc2_packed;
    FHEBlockTensor pre_before_bts;
    FHEBlockTensor pre;
    FHEBlockTensor grad_Wc1_packed;
    FHEBlockTensor gradient_x2;
};

inline FHEBlockTensor classifier_broadcast_binary_logits(FHEAdapterContext &adapter,
    const FHEBlockTensor &logits, int batch, const PhantomBootstrappingCallbacks &callbacks)
{
    require_single_ciphertext(logits, "classifier_broadcast_binary_logits");
    const int slots=static_cast<int>(adapter.require_ckks_encoder().slot_count());
    if (batch<=0 || (slots/2)%batch)
        throw std::invalid_argument("Invalid binary classifier batch layout");
    const double gain=callbacks.gain_compensated_bts ? 0.5 : 1.0;
    std::vector<double> mask(slots,0.0);
    for (int b=0;b<batch;++b) mask[b]=mask[slots/2+b]=gain;
    auto out=logits;
    PhantomPlaintext plain;
    encode_values(adapter,mask,logits.cts.front().params_id(),logits.cts.front().scale(),plain);
    out.cts.front()=multiply_plain_and_rescale(adapter,logits.cts.front(),plain,"classifier_logits_select");
    for (int rotation=batch;rotation<slots/2;rotation*=2) {
        PhantomCiphertext shifted;
        adapter.require_evaluator().rotate_vector(out.cts.front(),-rotation,adapter.require_galois_keys(),shifted);
        adapter.require_evaluator().add_inplace(out.cts.front(),shifted);
        
    }
    refresh_tensor_metadata(adapter.require_context(),adapter.level_map,out);
    align_down_to(adapter.require_context(), adapter.level_map, out, 0, out.name);
    auto adjusted=callbacks;
    if (gain!=1.0) adjusted.ordinary_bts=callbacks.gain_compensated_bts(gain);
    return ordinary_bts(adapter,out,adjusted,"classifier_logits_broadcast",logits.level,"classifier_logits_broadcast");
}

inline FHEBlockTensor classifier_w_c1_gradient_as_parameter_packing(const FHEBlockTensor &grad_Wc1_packed,
                                                                    const FHEBlockTensor &Wc1,
                                                                    const std::string &out_name = "grad_Wc1_parameter_packing")
{
    if (grad_Wc1_packed.cts.size() != Wc1.cts.size())
    {
        throw std::runtime_error("classifier_w_c1_gradient_as_parameter_packing ciphertext count mismatch");
    }
    if (Wc1.packing != PackingType::RepeatedColumn)
    {
        throw std::runtime_error("classifier_w_c1_gradient_as_parameter_packing requires Wc1 RepeatedColumn packing");
    }
    FHEBlockTensor out = grad_Wc1_packed;
    out.name = out_name;
    out.logical_shape = Wc1.logical_shape;
    out.packing = Wc1.packing;
    out.is_complex = Wc1.is_complex;

    return out;
}

inline FHEBlockTensor classifier_w_c2_gradient_as_parameter_packing(const FHEBlockTensor &grad_Wc2_packed,
                                                                    const FHEBlockTensor &Wc2,
                                                                    const std::string &out_name = "grad_Wc2_parameter_packing")
{
    if (grad_Wc2_packed.cts.size() != Wc2.cts.size())
    {
        throw std::runtime_error("classifier_w_c2_gradient_as_parameter_packing ciphertext count mismatch");
    }
    if (Wc2.packing != PackingType::InterleavedRow)
    {
        throw std::runtime_error("classifier_w_c2_gradient_as_parameter_packing requires Wc2 InterleavedRow packing");
    }
    FHEBlockTensor out = grad_Wc2_packed;
    out.name = out_name;
    out.logical_shape = Wc2.logical_shape;
    out.packing = Wc2.packing;
    out.is_complex = Wc2.is_complex;

    return out;
}

inline ClassifierForwardResult classifier_forward(FHEAdapterContext &adapter,
                                                  const FHEBlockTensor &X1,
                                                  const FHEBlockTensor &Wc1,
                                                  const FHEBlockTensor &Wc2,
                                                  const FHEBlockTensor &y_target,
                                                  const ClassifierForwardConfig &config,
                                                  FHEForwardCache &cache,
                                                  const std::string &block_name = "ClassifierForward")
{
    
    FHEBlockTensor x1 = X1;
    x1.name = "classifier_X1_extract_input_level6";
    align_down_to(adapter.require_context(), adapter.level_map, x1, config.x1_extract_input_level, x1.name);
    FHEBlockTensor x1_first_row = phantom_ext_copy_x(adapter,
                                                     x1,
                                                     config.c1_params.batch_size,
                                                     config.c1_params.row_X,
                                                     "classifier_X1_first_row_level5",
                                                     block_name + "_extract_first_row");
    assert_level(x1_first_row, config.x1_input_level, "classifier_X1_first_row");

    FHEBlockTensor wc1 = Wc1;
    wc1.name = "Wc1_level5";
    align_down_to(adapter.require_context(), adapter.level_map, wc1, config.x1_input_level, wc1.name);

    ClassifierForwardResult result;
    {
        result.X1Wc1 = phantom_ccmm_class_1_from_ext_batch(adapter,
                                                           x1_first_row,
                                                           wc1,
                                                           config.c1_params,
                                                           "X1Wc1",
                                                           block_name + "_CCMM_C1");
        assert_level(result.X1Wc1, config.x1w_c1_level, "X1Wc1");
    }
    if (!config.bias_c1.empty()) {
        if (static_cast<int>(config.bias_c1.size()) != config.c1_params.col_W)
            throw std::runtime_error("Classifier C1 bias size mismatch");
        const int slots = static_cast<int>(adapter.require_ckks_encoder().slot_count());
        for (size_t i = 0; i < result.X1Wc1.cts.size(); ++i) {
            std::vector<double> values(slots);
            for (int slot = 0; slot < slots; ++slot) {
                const int feature = (slot % (slots / 2)) / config.c1_params.batch_size;
                if (feature < config.c1_params.col_W)
                    values[slot] = config.bias_c1.at(feature);
            }
            PhantomPlaintext p;
            encode_values(adapter, values, result.X1Wc1.cts[i].params_id(),
                              result.X1Wc1.cts[i].scale(), p);
            adapter.require_evaluator().add_plain_inplace(result.X1Wc1.cts[i], p);
        }
    }
    result.tanh_classifier = rboot_fb(adapter,
                                      result.X1Wc1,
                                      config.callbacks,
                                      "tanh",
                                      "tanh_classifier",
                                      config.x1w_c1_level,
                                      config.tanh_classifier_level,
                                      block_name + "_Tanh_RBOOT");
    result.tanh_classifier.logical_shape = result.X1Wc1.logical_shape;
    result.tanh_classifier.packing = PackingType::InterleavedRow;
    result.tanh_classifier.is_complex = false;

    FHEBlockTensor wc2 = Wc2;
    align_down_to(adapter.require_context(), adapter.level_map, wc2, config.tanh_classifier_level, wc2.name);
    {
        result.X1Wc2 = phantom_ccmm_class_2_batch(adapter,
                                                  result.tanh_classifier,
                                                  wc2,
                                                  config.c2_params,
                                                  "X1Wc2",
                                                  block_name + "_CCMM_C2");
        assert_level(result.X1Wc2, config.x1w_c2_level, "X1Wc2");
    }
    if (!config.bias_c2.empty()) {
        if (config.bias_c2.size() != 2 || result.X1Wc2.cts.size() != 1)
            throw std::runtime_error("Classifier C2 bias packing mismatch");
        const int slots = static_cast<int>(adapter.require_ckks_encoder().slot_count());
        std::vector<double> values(slots);
        for (int slot = 0; slot < slots; ++slot)
            values[slot] = config.bias_c2[slot / (slots / 2)];
        PhantomPlaintext p;
        encode_values(adapter, values, result.X1Wc2.cts.front().params_id(),
                          result.X1Wc2.cts.front().scale(), p);
        adapter.require_evaluator().add_plain_inplace(result.X1Wc2.cts.front(), p);
    }
    result.X1Wc2=classifier_broadcast_binary_logits(adapter,result.X1Wc2,
        config.c2_params.batch_size,config.callbacks);
    result.y_pred = config.probability_backend(adapter, result.X1Wc2,
        config.bts2_stored_level, block_name + "_probabilities");
    assert_level(result.y_pred, config.bts2_stored_level, "y_pred");
    FHEBlockTensor y_ref = y_target;
    align_down_to(adapter.require_context(), adapter.level_map, y_ref, config.bts2_stored_level, y_ref.name);
    result.y_lr = fhe_subtract(adapter, result.y_pred, y_ref, "y_lr", block_name + "_y_lr_subtraction");
    assert_level(result.y_lr, config.bts2_stored_level, "y_lr");

    return result;
}

inline ClassifierBackwardResult classifier_backward(FHEAdapterContext &adapter,
                                                    const FHEBlockTensor &y_lr,
                                                    const FHEBlockTensor &tanh_classifier,
                                                    const FHEBlockTensor &Wc2,
                                                    const FHEBlockTensor &Wc1,
                                                    const FHEBlockTensor &classifier_input_X1,
                                                    const ClassifierBackwardConfig &config,
                                                    FHEBackwardCache &cache,
                                                    const std::string &block_name = "ClassifierBackward")
{
    
    FHEBlockTensor y_for_w2 = y_lr;
    align_down_to(adapter.require_context(), adapter.level_map, y_for_w2, config.grad_Wc2_input_level, y_for_w2.name);
    FHEBlockTensor tanh_for_w2 = tanh_classifier;
    align_down_to(adapter.require_context(), adapter.level_map, tanh_for_w2, config.grad_Wc2_input_level, tanh_for_w2.name);

    ClassifierBackwardResult result;
    result.grad_Wc2_packed = back_class_2_batch(adapter, y_for_w2, tanh_for_w2, "grad_Wc2_packed");
    assert_level(result.grad_Wc2_packed, config.grad_Wc2_level, "grad_Wc2_packed");

    FHEBlockTensor y_for_pre = y_lr;
    FHEBlockTensor tanh_for_pre = tanh_classifier;
    FHEBlockTensor wc2_for_pre = Wc2;
    align_down_to(adapter.require_context(), adapter.level_map, y_for_pre, config.pre_input_level, y_for_pre.name);
    align_down_to(adapter.require_context(), adapter.level_map, tanh_for_pre, config.pre_input_level, tanh_for_pre.name);
    align_down_to(adapter.require_context(), adapter.level_map, wc2_for_pre, config.pre_input_level, wc2_for_pre.name);
    {
        result.pre_before_bts = phantom_pre_compute_back_class_1(adapter,
                                                                 y_for_pre,
                                                                 wc2_for_pre,
                                                                 tanh_for_pre,
                                                                 config.cpu_params,
                                                                 "pre_before_BTS3",
                                                                 block_name + "_pre_compute");
        assert_level(result.pre_before_bts, config.pre_level, "pre_before_BTS3");
    }
    result.pre = ordinary_bts(adapter,
                              result.pre_before_bts,
                              config.callbacks,
                              "pre",
                              config.bts3_stored_level,
                              block_name + "_BTS3");

    FHEBlockTensor x_for_w1 = classifier_input_X1;
    align_down_to(adapter.require_context(), adapter.level_map, x_for_w1, config.grad_Wc1_x_input_level, x_for_w1.name);
    {
        result.grad_Wc1_packed = phantom_back_class_1_batch(adapter,
                                                            x_for_w1,
                                                            result.pre,
                                                            config.cpu_params,
                                                            "grad_Wc1_packed",
                                                            block_name + "_grad_Wc1");
        assert_level(result.grad_Wc1_packed, config.grad_Wc1_level, "grad_Wc1_packed");
    }
    FHEBlockTensor wc1_for_x2 = Wc1;
    align_down_to(adapter.require_context(), adapter.level_map, wc1_for_x2, config.grad_x2_input_level, wc1_for_x2.name);
    FHEBlockTensor pre_for_x2 = result.pre;
    align_down_to(adapter.require_context(), adapter.level_map, pre_for_x2, config.grad_x2_input_level, pre_for_x2.name);
    {
        result.gradient_x2 = phantom_gradient_x2(adapter,
                                                 wc1_for_x2,
                                                 pre_for_x2,
                                                 config.cpu_params,
                                                 "gradient_x2",
                                                 block_name + "_gradient_x2");
        assert_level(result.gradient_x2, config.grad_x2_level, "gradient_x2");
    }

    if (config.write_cache)
    {
        cache.grad_Wc2_packed = result.grad_Wc2_packed;
        cache.grad_Wc1_packed = result.grad_Wc1_packed;
        cache.gradient_x2 = result.gradient_x2;
        cache_write(cache, "grad_Wc2_packed", result.grad_Wc2_packed, "packed Wc2 gradient", "AdamW Wc2");
        cache_write(cache, "grad_Wc1_packed", result.grad_Wc1_packed, "packed Wc1 gradient", "AdamW Wc1");
        cache_write(cache, "gradient_x2", result.gradient_x2, "classifier gradient to transformer", "gradient_LN2");
    }
    return result;
}

}
