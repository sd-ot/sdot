#pragma once

#include "type_config.h"
#include "Arch.h"

namespace sdot {

template<class Arch,class T>
struct SimdAlig;

template<class T> struct SimdAlig<NOARCH,T   > { static constexpr ST value =  1; };

template<       > struct SimdAlig<SSE2  ,FP32> { static constexpr ST value =  4; };
template<       > struct SimdAlig<AVX2  ,FP32> { static constexpr ST value =  8; };
template<       > struct SimdAlig<AVX512,FP32> { static constexpr ST value = 16; };

template<       > struct SimdAlig<SSE2  ,U32 > { static constexpr ST value =  4; };
template<       > struct SimdAlig<AVX2  ,U32 > { static constexpr ST value =  8; };
template<       > struct SimdAlig<AVX512,U32 > { static constexpr ST value = 16; };

template<       > struct SimdAlig<SSE2  ,FP64> { static constexpr ST value =  2; };
template<       > struct SimdAlig<AVX2  ,FP64> { static constexpr ST value =  4; };
template<       > struct SimdAlig<AVX512,FP64> { static constexpr ST value =  8; };

template<       > struct SimdAlig<SSE2  ,U64 > { static constexpr ST value =  2; };
template<       > struct SimdAlig<AVX2  ,U64 > { static constexpr ST value =  4; };
template<       > struct SimdAlig<AVX512,U64 > { static constexpr ST value =  8; };

}
