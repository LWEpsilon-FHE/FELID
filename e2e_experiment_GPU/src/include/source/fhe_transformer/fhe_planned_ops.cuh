#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "source/fhe_transformer/fhe_ops.cuh"

namespace fhe_transformer {

using TensorUnaryBackend = std::function<FHEBlockTensor(FHEAdapterContext &, const FHEBlockTensor &, const PlannedOpSpec &)>;

}
