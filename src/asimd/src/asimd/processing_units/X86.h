#pragma once

#include "X86Features.h"

namespace asimd {
namespace processing_units {

/**
*/
template<int ptr_size,class... Features>
struct X86 : FeatureSet<Features...> {
    static std::string name           () { return "X86<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }

    std::size_t        L1_cache_size; ///<
    std::size_t        L2_cache_size; ///<
    std::size_t        nb_cores;      ///<
};

} // namespace processing_units
} // namespace asimd
