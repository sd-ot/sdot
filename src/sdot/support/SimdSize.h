#pragma once

#include "type_config.h"
#include "Arch.h"

namespace sdot {

template<class Arch,class T>
struct SimdSize;

template<class T> struct SimdSize<NOARCH,T   > { static constexpr ST value =  1; };

template<       > struct SimdSize<SSE2  ,FP32> { static constexpr ST value =  4; };
template<       > struct SimdSize<AVX2  ,FP32> { static constexpr ST value =  8; };
template<       > struct SimdSize<AVX512,FP32> { static constexpr ST value = 16; };

template<       > struct SimdSize<SSE2  ,U32 > { static constexpr ST value =  4; };
template<       > struct SimdSize<AVX2  ,U32 > { static constexpr ST value =  8; };
template<       > struct SimdSize<AVX512,U32 > { static constexpr ST value = 16; };

template<       > struct SimdSize<SSE2  ,FP64> { static constexpr ST value =  2; };
template<       > struct SimdSize<AVX2  ,FP64> { static constexpr ST value =  4; };
template<       > struct SimdSize<AVX512,FP64> { static constexpr ST value =  8; };

template<       > struct SimdSize<SSE2  ,U64 > { static constexpr ST value =  2; };
template<       > struct SimdSize<AVX2  ,U64 > { static constexpr ST value =  4; };
template<       > struct SimdSize<AVX512,U64 > { static constexpr ST value =  8; };

}
