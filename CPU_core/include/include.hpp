#ifndef INCLUDE_H
#define INCLUDE_H

#pragma once

//intel hexl library
// #include "hexl/hexl.hpp"

//SEAL library
#include "seal/seal.h"

//C++
#include <iostream>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <functional>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <string>
#include <memory>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <iomanip>

#include <chrono>

//source code
#include "source/Packing.hpp"
#include "source/PCMM.hpp"
#include "source/CCMM.hpp"
#include "source/CCMM_classifier.hpp"
#include "source/Layernorm.hpp"
#include "source/Layernorm_v2.hpp"
#include "source/Softmax_classifier.hpp"

#include "source/CCMM_back.hpp"
#include "source/back_classifier.hpp"
#include "source/back_ffn.hpp"
#include "source/back_att.hpp"
#include "source/back_lora.hpp"


//test
#include "test/test_SEAL_ckks.hpp"

#include "test/bootstrapping/test_bootstrapping.hpp"

#include "test/test_packing.hpp"

#include "test/test_PCMM.hpp"

#include "test/test_CCMM_LoraAB.hpp"

#include "test/test_CCMM_kernel.hpp"

#include "test/test_CCMM_V.hpp"

#include "test/test_kernelV_bsgs_selfoutput.hpp"

#include "test/test_CCMM_class.hpp"

#include "test/test_layernorm.hpp"

#include "test/test_softmax_class.hpp"



#endif
