#pragma once

#include "GenericFeatures.h"
#include "FeatureSet.h"

namespace asimd {
namespace processing_units {

/**
*/
template<int ptr_size = 8 * sizeof( void * ),class... Features>
struct NvidiaGpu : FeatureSet<Features...> {
    static std::string name() { return "NvidiaGpu<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace processing_units
} // namespace asimd
