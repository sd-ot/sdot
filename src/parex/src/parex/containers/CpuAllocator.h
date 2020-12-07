#ifndef parex_CpuMemory_H
#define parex_CpuMemory_H

#include <asimd/processing_units/LargestCpu.h>
#include <asimd/SimdSize.h>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <array>

#include "../data/TypeInfo.h"

namespace parex {

/**

 */
struct CpuAllocator {
    template<class T> struct         Alignment   { static constexpr std::size_t value = asimd::SimdSize<T,asimd::processing_units::LargestCpu>::value; };

    static constexpr bool            cpu         = true;
    using                            I           = std::size_t;

    /**/                             CpuAllocator() : used( 0 ) {}
    template<class T> void           deallocate  ( T *ptr, I count );
    template<class T> T*             allocate    ( I count );

    static CpuAllocator              local;
    std::atomic<I>                   used;
};

template<>
struct TypeInfo<CpuAllocator> {
    static std::string name() {
        return "parex::CpuAllocator";
    }
};

template<class T>
void copy_memory_values( const CpuAllocator &dst_alloc, T *dst_data, const CpuAllocator &src_alloc, const T *src_data, std::size_t len );

template<class T,int n,class F>
void get_memory_values( CpuAllocator &, CpuAllocator &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );


} // namespace parex

#include "CpuAllocator.tcc"

#endif // parex_CpuMemory_H
