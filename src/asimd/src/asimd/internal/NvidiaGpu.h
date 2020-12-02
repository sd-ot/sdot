#pragma once

#include "FeatureSet.h"

namespace asimd {
namespace processing_units {

/**
*/
template<int ptr_size = 8 * sizeof( void * ),class... Features>
struct NvidiaGpu : FeatureSet<Features...> {
    static std::string name           () { return "NvidiaGpu<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }

    std::size_t        L1_cache_size; ///<
    std::size_t        L2_cache_size; ///<
    std::size_t        nb_cores;      ///< nb SM
};

} // namespace processing_units
} // namespace asimd
