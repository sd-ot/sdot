#pragma once

#include <type_traits>
#include <cstdint>
#include <string>

namespace asimd {
namespace InstructionSet {

template<int ptr_size=8*sizeof(void *)>
struct Generic {
    using size_t = typename std::conditional<ptr_size==64,std::uint64_t,std::uint32_t>::type;
    static std::string name() { return "Generic<" + std::to_string( ptr_size ) + ">"; }
    enum { cpu = 1 };

    template<class T>
    struct MaxSimdSize {
        enum { v = 512 / 8 / sizeof( T ) };
        enum { value = v ? v : 1 };
    };

    template<class T>
    struct MaxSimdAlig {
        enum { value = MaxSimdSize<T>::value };
    };
};

template<int ptr_size=8*sizeof(void *)>
struct SSE2 : Generic<ptr_size> {
    static std::string name() { return "SSE2<" + std::to_string( ptr_size ) + ">"; }
    enum { sse2 = 1 };
};

template<int ptr_size=8*sizeof(void *)>
struct AVX : SSE2<ptr_size> {
    static std::string name() { return "AVX<" + std::to_string( ptr_size ) + ">"; }
    enum { avx = 1 };
};

template<int ptr_size=8*sizeof(void *)>
struct AVX2 : AVX<ptr_size> {
    static std::string name() { return "AVX2<" + std::to_string( ptr_size ) + ">"; }
    enum { avx2 = 1 };
};

template<int ptr_size=8*sizeof(void *)>
struct AVX512 : AVX2<ptr_size> {
    static std::string name() { return "AVX512<" + std::to_string( ptr_size ) + ">" ; }
    enum { avx512 = 1 };
};

template<int ptr_size=8*sizeof(void *)>
struct Gpu {
    using size_t = typename std::conditional<ptr_size==64,std::uint64_t,std::uint32_t>::value;
    static std::string name() { return "Gpu<" + std::to_string( ptr_size ) + ">"; }
    enum { gpu = 1 };
};

#if defined( __AVX512F__ )
using Native = AVX512<>;
#elif defined( __AVX2__ )
using Native = AVX2<>;
#elif defined( __SSE2__ )
using Native = SSE2<>;
#else
using Native = Generic<>;
#endif

} // namespace InstructionSet
} // namespace asimd
