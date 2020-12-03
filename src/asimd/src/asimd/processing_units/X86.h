#pragma once

#include "GenericFeatures.h"
#include "X86Features.h"
#include "FeatureSet.h"

namespace asimd {
namespace processing_units {

/**
*/
template<int ptr_size,class... Features>
struct X86 : FeatureSet<Features...> {
    static std::string name() { return "X86<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace processing_units
} // namespace asimd
