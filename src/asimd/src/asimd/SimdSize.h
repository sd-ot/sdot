#pragma once

#ifndef SDOT_SIMD_VEC_H
#error SimdSize should be included via SimdVec
#endif

#include "SimdVec.h"
#include "Arch.h"

namespace asimd {

template<class T,class Arch=Arch::Native>
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
