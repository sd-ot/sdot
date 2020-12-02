#pragma once

#include "internal/NvidiaGpu.h"
#include "internal/X86.h"

namespace asimd {
namespace processing_units {

// -------------------------- Native --------------------------
#ifdef __x86_64__
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
        , Features::SSE2
    #endif
>;
#endif // __x86_64__

} // namespace processing_units
} // namespace asimd
