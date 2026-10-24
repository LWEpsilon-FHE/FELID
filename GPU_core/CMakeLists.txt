cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 90)
endif()
project(FELID_GPU LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(OpenMP REQUIRED)
find_path(NTL_INCLUDE_DIR NTL/RR.h REQUIRED)
find_library(NTL_LIBRARY ntl REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
file(GLOB_RECURSE FELID_GPU_SOURCES CONFIGURE_DEPENDS thirdparty/phantom-fhe/src/*.cu src/include/source/bootstrapping/*.cu)
list(APPEND FELID_GPU_SOURCES src/include/source/ckks_evaluator_parallel.cu)
add_library(felid_gpu STATIC ${FELID_GPU_SOURCES})
set_target_properties(felid_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(felid_gpu PUBLIC thirdparty/phantom-fhe/include src/include src/include/source/bootstrapping src/include/source/bootstrapping/common ${NTL_INCLUDE_DIR})
target_compile_definitions(felid_gpu PUBLIC PHANTOM_USE_CUDA_PTX)
target_compile_options(felid_gpu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(felid_gpu PUBLIC OpenMP::OpenMP_CXX ${NTL_LIBRARY} ${GMP_LIBRARY} m pthread)
add_executable(core_test src/test.cu)
set_target_properties(core_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(core_test PRIVATE felid_gpu)
configure_file(src/data/Hermite4.txt Hermite4.txt COPYONLY)

enable_testing()
foreach(case IN ITEMS
    PCMM_batch_reference_test
    Layernorm_affine_test
    Packing_test
    Packing_batch_test
    vec_left_rot_test
    Internal_rot_test
    PCMM_test
    PCMM_batch_test
    PCMM_batch_complex_test
    CCMM_loraAB_test
    CCMM_loraAB_complex_test
    CCMM_KernelQK_test
    CCMM_Vtest
    KernelQK_V_bsgs_selfoutput_test
    CCMM_class_test
    Layernorm_test
    Layernorm_boot_selfoutput_test
    Softmax_test
    Softmax_boot_test
)
    add_test(NAME ${case} COMMAND core_test ${case})
endforeach()

set_target_properties(core_test PROPERTIES OUTPUT_NAME test)
