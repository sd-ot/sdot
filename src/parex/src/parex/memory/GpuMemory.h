#ifndef asimd_GpuMemory_H
#define asimd_GpuMemory_H

#include <cuda_runtime.h>
#include "CpuMemory.h"

//// nsmake lib_name cudart

namespace asimd {

/** std allocator for aligned memory */
struct GpuMemory {
    template<class T> struct Alignment { static constexpr std::size_t value = 256; };
    static constexpr bool    gpu       = true;
    using                    TI        = std::size_t;

    /**/                     GpuMemory ( int num_gpu = 0 );

    template<class T> void   deallocate( T *ptr, TI count );
    template<class T> T*     allocate  ( TI count );
    template<class T> T      value     ( const T *ptr );

    int                      num_gpu;  ///<
    std::atomic<TI>          used;     ///<
};

template<class T,int n,class F> void get_memory_values( GpuMemory &dst, GpuMemory &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( CpuMemory &dst, GpuMemory &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( GpuMemory &dst, CpuMemory &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace asimd

#include "GpuMemory.tcc"

#endif // asimd_GpuMemory_H
