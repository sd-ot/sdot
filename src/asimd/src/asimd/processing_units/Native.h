#pragma once

#include "NvidiaGpu.h"
#include "X86.h"

namespace asimd {
namespace processing_units {

// -------------------------- Native --------------------------
#if ( defined(_M_IX86) || defined(__i386__) || defined(_M_X64) || defined(__x86_64__) )
using Native = X86< 8 * sizeof( void * )
    #ifdef __AVX512F__
        , Features::AVX512
    #endif
    #ifdef __AVX2__
        , Features::AVX2
    #endif
    #ifdef __AVX__
       , Features::AVX
    #endif
    #ifdef __SSE2__
        , features::SSE2
    #endif
>;
#endif // x86

} // namespace processing_units
} // namespace asimd
