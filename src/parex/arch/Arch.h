#pragma once

namespace parex {
namespace Arch {

struct Generic { static const char *name() { return "Generic"; } };
struct AVX512  { static const char *name() { return "AVX512" ; }  enum { cpu = 1, sse2 = 1, avx = 1, avx2 = 1, avx512 = 1 }; };
struct AVX2    { static const char *name() { return "AVX2"   ; }  enum { cpu = 1, sse2 = 1, avx = 1, avx2 = 1 }; };
struct SSE2    { static const char *name() { return "SSE2"   ; }  enum { cpu = 1, sse2 = 1 }; };
struct Gpu     { static const char *name() { return "Gpu"    ; }  enum { gpu = 1 }; };

#if defined( __AVX512F__ )
using Native = AVX512;
#elif defined( __AVX2__ )
using Native = AVX2;
#elif defined( __SSE2__ )
using Native = SSE2;
#else
using Native = Generic;
#endif

} // namespace Arch
} // namespace parex
