#pragma once

#ifndef SDOT_SIMD_VEC_H
#error SimdSize should be included via SimdVec
#endif

#include "../CpuArch.h"
#include "SimdVec.h"

template<class T,class Arch=CpuArch::Native>
struct SimdSize {
    enum { value = 1 };
};

template<class T,class Arch>
struct SimdAlig {
    enum { value = SimdSize<T,Arch>::value };
};

#define DECL_SIMD_SIZE( T, ARCH, SIZE ) \
    template<> struct SimdSize<T,ARCH> { enum { value = SIZE }; }

