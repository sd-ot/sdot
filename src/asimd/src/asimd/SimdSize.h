#pragma once

#include "InstructionSet.h"

namespace asimd {

template<class T,class Arch=InstructionSet::Native>
struct SimdSize {
    enum { value = 1 };
};

template<class T,class Arch>
struct SimdAlig {
    enum { value = SimdSize<T,Arch>::value };
};

#define DECL_SIMD_SIZE( T, ARCH, SIZE ) \
    template<int ptr_size> struct SimdSize<T,ARCH<ptr_size>> { enum { value = SIZE }; }

} // namespace asimd
