#include <cpu_features/cpu_features_macros.h>
#include "CpuMemory.h"
#include <thread>
#include "X86Proc.h"

#ifdef CPU_FEATURES_ARCH_X86
#include <cpu_features/cpuinfo_x86.h>
#endif


namespace parex {
namespace hardware_information {

std::size_t X86Proc::ptr_size() const {
    return ptr_size_;
}

std::string X86Proc::name() const {
    return "X86Proc";
}

void X86Proc::get_locals( std::vector<std::unique_ptr<ProcessingUnit>> &pus, std::vector<std::unique_ptr<Memory>> &memories ) {
    #if defined(CPU_FEATURES_ARCH_X86)
    std::unique_ptr<X86Proc> cpu = std::make_unique<X86Proc>();
    cpu->ptr_size_ = 8 * sizeof( void * );

    // instructions
    cpu_features::X86Info xi = cpu_features::GetX86Info();
    if ( xi.features.avx512f ) cpu->features[ "AVX512" ];
    if ( xi.features.avx2    ) cpu->features[ "AVX2"   ];
    if ( xi.features.avx     ) cpu->features[ "AVX"    ];
    if ( xi.features.sse2    ) cpu->features[ "SSE2"   ];

    // caches
    cpu_features::CacheInfo ci = cpu_features::GetX86CacheInfo();
    for( int num = 0; num < ci.size; ++num ) {
        if ( ci.levels[ num ].cache_type == cpu_features::CPU_FEATURE_CACHE_DATA || ci.levels[ num ].cache_type == cpu_features::CPU_FEATURE_CACHE_UNIFIED ) {
            cpu->features[ "L" + std::to_string( ci.levels[ num ].level ) + "Cache" ] = std::string( "{ " ) +
                ".amount = " + std::to_string( ci.levels[ num ].cache_size ) + ", " +
                ".ways = " + std::to_string( ci.levels[ num ].ways ) + ", " +
                ".line_size = " + std::to_string( ci.levels[ num ].line_size ) +
             " }";
        }
    }

    // multithreading
    if ( int n = std::thread::hardware_concurrency() )
        cpu->features[ "Multithread" ] = std::to_string( n );

    // memory
    std::unique_ptr<CpuMemory> mem = std::make_unique<CpuMemory>();
    mem->allocator = &CpuAllocator::local;
    mem->is_local = true;

    mem->register_link( {
        .processing_unit = cpu.get(),
        .bandwidth = 90e9
    } );

    // register
    memories.push_back( std::move( mem ) );
    pus.push_back( std::move( cpu ) );
    #endif
}

} // namespace hardware_information
} // namespace parex
