#pragma once

#include "processing_units.h"

namespace asimd {

template<class T,class Is=processing_units::Native>
struct SimdSize {
    static constexpr int value = Is::template SimdSize<T>::value;
};

} // namespace asimd
