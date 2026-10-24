cmake_minimum_required(VERSION 3.20)
project(FELID_CPU LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SEAL_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(SEAL_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(SEAL_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(SEAL_USE_INTEL_HEXL OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SEAL_BUILD_SEAL_C OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/SEAL-4.1-bs)
find_path(NTL_INCLUDE_DIR NTL/RR.h REQUIRED)
find_library(NTL_LIBRARY ntl REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
find_package(OpenMP REQUIRED)
file(GLOB_RECURSE FELID_CPU_SOURCES CONFIGURE_DEPENDS include/source/bootstrapping/*.cpp)
add_library(felid_cpu STATIC ${FELID_CPU_SOURCES})
target_include_directories(felid_cpu PUBLIC include include/source include/source/bootstrapping include/source/bootstrapping/common ${NTL_INCLUDE_DIR})
target_link_libraries(felid_cpu PUBLIC SEAL::seal OpenMP::OpenMP_CXX ${NTL_LIBRARY} ${GMP_LIBRARY} m pthread)
add_executable(core_test test.cpp)
target_link_libraries(core_test PRIVATE felid_cpu)

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
