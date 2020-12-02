#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <memory>
#include <array>

//// nsmake lib_name cudart

namespace asimd {

/** std allocator for aligned memory */
template<class T>
struct GpuAllocator : std::allocator<T> {
    template<class U> struct rebind      { using other = GpuAllocator<U>; };
    static constexpr size_t  alignment   = 256;
    static constexpr bool    gpu         = true;

    /**/                     GpuAllocator( int num_gpu = 0 ) : num_gpu( num_gpu ) {}
    template<class T2>       GpuAllocator( const GpuAllocator<T2> &that ) : num_gpu( that.num_gpu ) {}
    static void              deallocate  ( T *ptr, std::size_t = 0 ) { cudaFree( ptr ); }
    static T*                allocate    ( std::size_t count, const void* = 0 ) { T *res; cudaMalloc( &res, sizeof( T ) * count ); return res; }
    T                        value       ( const T *ptr ) const { T res; cudaMemcpy( &res, ptr, sizeof( T ), cudaMemcpyDeviceToHost ); return res; }

    template<class F> void   to_local    ( F &&f, std::array<T,n> ptrs ) const;

    int                      num_gpu;    ///<
};

} // namespace asimd
