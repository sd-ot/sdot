#pragma once

#include "internal/FeatureSet.h"
#include <cstdint>

namespace asimd {
namespace processing_units {

namespace Features {
    template<int size_in_bytes,class... Types> struct SimdOn {
        template<class T> struct SimdSize { static constexpr int value = FeatureSet<Types...>::template Has<T>::value ? size_in_bytes / sizeof( T ) : 1; };
        template<class T> struct SimdAlig { static constexpr int value = SimdSize<T>::value; };
    };

    #define ASIMD_CMON_TYPES float,double,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t
    // X86 features
    struct AVX512 : SimdOn<64,ASIMD_CMON_TYPES> { static std::string name() { return "AVX512"; } };
    struct AVX2   : SimdOn<32,ASIMD_CMON_TYPES> { static std::string name() { return "AVX2"  ; } };
    struct AVX    : SimdOn<32,ASIMD_CMON_TYPES> { static std::string name() { return "AVX"   ; } };
    struct SSE2   : SimdOn<16,ASIMD_CMON_TYPES> { static std::string name() { return "SSE2"  ; } };
    #undef ASIMD_CMON_TYPES
}

// -------------------------- X86 --------------------------
template<int ptr_size,class... Features>
struct X86 : FeatureSet<Features...> {
    static std::string name() { return "X86<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }
};

// -------------------------- Cuda --------------------------
template<int ptr_size = 8 * sizeof( void * ),class... Features>
struct NvidiaGpu : FeatureSet<Features...> {
    static std::string name() { return "NvidiaGpu<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }
};

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
