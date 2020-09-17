#pragma once

#include <type_traits>
#include <cstdint>

#ifdef __CUDACC__
#include "GpuInfo.h"
#endif // __CUDACC__

namespace sdot {

namespace MachineArch {

// Arch types
struct GenericCpu          { std::size_t L1_size = 64 * 1024; };
struct AVX512 : GenericCpu { enum { cpu = 1, sse2 = 1, avx = 1, avx2 = 1, avx512 = 1 }; };
struct AVX2   : GenericCpu { enum { cpu = 1, sse2 = 1, avx = 1, avx2 = 1 }; };
struct SSE2   : GenericCpu { enum { cpu = 1, sse2 = 1 }; };

#ifdef __CUDACC__
struct Gpu    : GpuInfo    { enum { gpu = 1 }; };
#endif // __CUDACC__

// Native
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

// OnGpu
template<class Arch,class Enable=void> struct OnGpu;
template<class Arch> struct OnGpu<Arch,typename std::enable_if<Arch::cpu>::type> { enum { value = 0 }; };
template<class Arch> struct OnGpu<Arch,typename std::enable_if<Arch::gpu>::type> { enum { value = 1 }; };

} // namespace sdot
