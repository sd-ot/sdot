#include <cpu_features/cpu_features_macros.h>
#include <sstream>
#include "X86.h"

#ifdef CPU_FEATURES_ARCH_X86
#include <cpu_features/cpuinfo_x86.h>
#endif

namespace parex {
namespace hardware_information {

void X86::asimd_init( std::ostream &os, const std::string &var_name, const std::string &sp ) const {
    for( const auto &p : features )
        if ( p.second.size() )
            os << sp << var_name << ".value<" << p.first << ">() = " << p.second << ";\n";
}

std::string X86::asimd_name() const {
    std::ostringstream ss;
    ss << "X86<" << ptr_size();
    for( const auto &p : features )
        ss << "," << p.first;
    ss << ">";
    return ss.str();
}

std::size_t X86::ptr_size() const {
    return ptr_size_;
}

std::unique_ptr<X86> X86::local() {
    #if defined(CPU_FEATURES_ARCH_X86)
    std::unique_ptr<X86> res = std::make_unique<X86>();
    res->ptr_size_ = 8 * sizeof( void * );

    cpu_features::X86Info xi = cpu_features::GetX86Info();
    if ( xi.features.avx512f ) res->features[ "AVX512" ];
    if ( xi.features.avx2    ) res->features[ "AVX2"   ];
    if ( xi.features.avx     ) res->features[ "AVX"    ];
    if ( xi.features.sse2    ) res->features[ "SSE2"   ];

    cpu_features::CacheInfo ci = cpu_features::GetX86CacheInfo();
    for( int num = 0; num < ci.size; ++num )
        if ( ci.levels[ num ].cache_type == cpu_features::CPU_FEATURE_CACHE_DATA || ci.levels[ num ].cache_type == cpu_features::CPU_FEATURE_CACHE_UNIFIED )
            res->features[ "L" + std::to_string( ci.levels[ num ].level ) + "Cache" ] = std::string( "{ " ) +
                    ".amount = " + std::to_string( ci.levels[ num ].cache_size ) + ", " +
                    ".ways = " + std::to_string( ci.levels[ num ].ways ) + ", " +
                    ".line_size = " + std::to_string( ci.levels[ num ].line_size ) +
                    " }";

    return res;
    #else
    return {};
    #endif
}

} // namespace hardware_information
} // namespace parex
