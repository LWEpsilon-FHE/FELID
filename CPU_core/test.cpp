#include <iostream>
#include "include.hpp"

using namespace std;




#include <exception>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[] = {
        {"PCMM_batch_reference_test", PCMM_batch_reference_test},
        {"Layernorm_affine_test", Layernorm_affine_test},
        {"Packing_test", Packing_test},
        {"Packing_batch_test", Packing_batch_test},
        {"vec_left_rot_test", vec_left_rot_test},
        {"Internal_rot_test", Internal_rot_test},
        {"PCMM_test", PCMM_test},
        {"PCMM_batch_test", PCMM_batch_test},
        {"PCMM_batch_complex_test", PCMM_batch_complex_test},
        {"CCMM_loraAB_test", CCMM_loraAB_test},
        {"CCMM_loraAB_complex_test", CCMM_loraAB_complex_test},
        {"CCMM_KernelQK_test", CCMM_KernelQK_test},
        {"CCMM_Vtest", CCMM_Vtest},
        {"KernelQK_V_bsgs_selfoutput_test", KernelQK_V_bsgs_selfoutput_test},
        {"CCMM_class_test", CCMM_class_test},
        {"Layernorm_test", Layernorm_test},
        {"Layernorm_boot_selfoutput_test", Layernorm_boot_selfoutput_test},
        {"Softmax_test", Softmax_test},
        {"Softmax_boot_test", Softmax_boot_test},
    };
    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (const auto& test : tests) std::cout << test.name << '\n';
        return 0;
    }
    std::vector<const Test*> selected;
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "all")) {
        for (const auto& test : tests) selected.push_back(&test);
    } else {
        for (int i = 1; i < argc; ++i) {
            const Test* match = nullptr;
            for (const auto& test : tests) if (test.name == std::string(argv[i])) match = &test;
            if (!match) { std::cerr << "Unknown test: " << argv[i] << '\n'; return 2; }
            selected.push_back(match);
        }
    }
    int failures = 0;
    for (const auto* test : selected) {
        std::cout << "[ RUN ] " << test->name << std::endl;
        try { test->run(); std::cout << "[ PASS ] " << test->name << std::endl; }
        catch (const std::exception& error) {
            ++failures;
            std::cerr << "[ FAIL ] " << test->name << ": " << error.what() << std::endl;
        }
    }
    std::cout << "Tests: " << selected.size() << ", failures: " << failures << std::endl;
    return failures ? 1 : 0;
}
