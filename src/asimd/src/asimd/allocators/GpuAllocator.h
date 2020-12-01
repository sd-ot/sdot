#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <memory>

//// nsmake lib_name cudart

namespace asimd {

/** std allocator for aligned memory */
template<class T>
struct GpuAllocator : std::allocator<T> {
    static constexpr size_t  alignment   = 256;
    template<class U> struct rebind      { using other = GpuAllocator<U>; };

    /**/                     GpuAllocator( int num_gpu = 0 ) : num_gpu( num_gpu ) {}
    static void              deallocate  ( T *ptr, std::size_t = 0 ) { cudaFree( ptr ); }
    static T*                allocate    ( std::size_t count, const void* = 0 ) { T *res; cudaMalloc( &res, sizeof( T ) * count ); return res; }
    T                        value       ( const T *ptr ) const { T res; cudaMemcpy( &res, ptr, sizeof( T ), cudaMemcpyDeviceToHost ); return res; }

    int                      num_gpu;    ///<
};

} // namespace asimd
