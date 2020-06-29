#pragma once

#include <cstdint>

namespace CpuArch {

struct Generic {};
struct AVX512  { enum { sse2 = 1, avx = 1, avx2 = 1, avx512 = 1 }; };
struct AVX2    { enum { sse2 = 1, avx = 1, avx2 = 1 }; };
struct SSE2    { enum { sse2 = 1 }; };

#if defined( __AVX512F__ )
using Native = AVX512;
#elif defined( __AVX2__ )
using Native = AVX2;
#elif defined( __SSE2__ )
using Native = SSE2;
#else
using Native = Generic;
#endif

}
