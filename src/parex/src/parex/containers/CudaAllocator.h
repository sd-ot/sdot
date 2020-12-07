#ifndef parex_GpuAllocator_H
#define parex_GpuAllocator_H

#include <cuda_runtime.h>
#include "CpuAllocator.h"

//// nsmake lib_name cudart

namespace parex {

/** std allocator for aligned memory */
struct GpuAllocator {
    template<class T> struct Alignment   { static constexpr std::size_t value = 256; };
    static constexpr bool    gpu         = true;
    using                    I           = std::size_t;

    /**/                     GpuAllocator( int num_gpu = 0 );

    template<class T> void   deallocate  ( T *ptr, I count );
    template<class T> T*     allocate    ( I count );
    template<class T> T      value       ( const T *ptr );

    int                      num_gpu;    ///<
    std::atomic<I>           used;       ///<
};

template<class T,int n,class F> void get_memory_values( GpuAllocator &dst, GpuAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( CpuAllocator &dst, GpuAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( GpuAllocator &dst, CpuAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace parex

#include "GpuAllocator.tcc"

#endif // parex_GpuAllocator_H
