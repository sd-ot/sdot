#pragma once

#include "InstructionSet.h"

namespace asimd {

template<class T,class Is=InstructionSet::Native>
struct SimdSize {
    static constexpr int value = Is::template SimdSize<T>::value;
};

} // namespace asimd
