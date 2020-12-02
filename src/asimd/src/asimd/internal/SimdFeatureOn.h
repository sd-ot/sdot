#pragma once

#include "FeatureSet.h"

namespace asimd {
namespace processing_units {
namespace Features {

/**
*/
template<int size_in_bytes,class... Types>
struct SimdFeatureOn {
    template<class T> struct SimdSize { static constexpr int value = FeatureSet<Types...>::template Has<T>::value ? size_in_bytes / sizeof( T ) : 1; };
    template<class T> struct SimdAlig { static constexpr int value = SimdSize<T>::value; };
};

} // namespace Features
} // namespace processing_units
} // namespace asimd
