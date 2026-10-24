#pragma once

#include <stdexcept>
#include <string>

#include "source/fhe_transformer/fhe_cache_utils.cuh"
#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

struct GeluParams {
    int expected_input_level = 3;
    int stored_input_level = 5;
    int stored_derivative_level = 4;
    int stored_output_level = 4;
};

struct GeluResult {
    FHEBlockTensor gelu_input;
    FHEBlockTensor gelu_derivative;
    FHEBlockTensor gelu_output;
};

inline GeluResult run_gelu_with_rboot(FHEAdapterContext &adapter,
                                      const FHEBlockTensor &input,
                                      const GeluParams &params,
                                      const PhantomBootstrappingCallbacks &callbacks,
                                      const std::string &step_name = "GELU_RBOOT")
{
    
    if (input.is_complex || input.packing != PackingType::InterleavedColumn)
    {
        throw std::runtime_error(step_name + " requires real InterleavedColumn input");
    }
    
    assert_level(input, params.expected_input_level, input.name);

    GeluResult result;

    if (callbacks.gelu_derivative_component)
    {
        
        result.gelu_derivative = callbacks.gelu_derivative_component(adapter,
                                                          input,
                                                          adapter.level_map.max_logical_level,
                                                          "gelu_derivative",
                                                          step_name + "_tanh_component");
    }
    else
    {
        result.gelu_derivative = rboot_fb(adapter,
                                    input,
                                    callbacks,
                                    "gelu_tanh",
                                    "gelu_derivative",
                                    params.expected_input_level,
                                    adapter.level_map.max_logical_level,
                                    step_name + "_tanh_component");
    }
    result.gelu_derivative.logical_shape = input.logical_shape;
    result.gelu_derivative.packing = input.packing;
    result.gelu_derivative.is_complex = false;
    assert_level(result.gelu_derivative, adapter.level_map.max_logical_level, result.gelu_derivative.name);

    if (callbacks.gelu_input_for_backward)
    {
        
        result.gelu_input = callbacks.gelu_input_for_backward(adapter,
                                                              input,
                                                              params.stored_input_level,
                                                              "gelu_input",
                                                              step_name + "_input_cache");
    }
    else
    {
        result.gelu_input = input;
        result.gelu_input.name = "gelu_input";
        if (result.gelu_input.level > params.stored_input_level)
        {
            align_down_to(adapter.require_context(), adapter.level_map, result.gelu_input, params.stored_input_level, result.gelu_input.name);
        }
        if (result.gelu_input.level < params.stored_input_level)
        {
            
            throw std::runtime_error(step_name + " cannot cache gelu_input at required backward level");
        }
    }
    result.gelu_input.logical_shape = input.logical_shape;
    result.gelu_input.packing = input.packing;
    result.gelu_input.is_complex = false;
    assert_level(result.gelu_input, params.stored_input_level, result.gelu_input.name);

    if (!callbacks.gelu_output_component)
    {
        throw std::runtime_error(step_name +
                                 " requires a GELU value backend. "
                                 "");
    }
    
    result.gelu_output = callbacks.gelu_output_component(adapter,
                                                            result.gelu_input,
                                                            result.gelu_derivative,
                                                            params.stored_output_level,
                                                            "gelu_output",
                                                            step_name + "_gelu_postprocess");
    result.gelu_output.logical_shape = input.logical_shape;
    result.gelu_output.packing = input.packing;
    result.gelu_output.is_complex = false;
    assert_level(result.gelu_output, params.stored_output_level, result.gelu_output.name);

    if(result.gelu_derivative.level>params.stored_derivative_level)
        align_down_to(adapter.require_context(),adapter.level_map,result.gelu_derivative,
            params.stored_derivative_level,result.gelu_derivative.name);
    assert_level(result.gelu_derivative,params.stored_derivative_level,result.gelu_derivative.name);
    return result;
}

}
