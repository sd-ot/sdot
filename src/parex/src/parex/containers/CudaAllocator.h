#ifndef parex_GpuAllocator_H
#define parex_GpuAllocator_H

#include "CpuAllocator.h"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

//// nsmake lib_name cudart

namespace parex {

/** std allocator for aligned memory */
struct CudaAllocator {
    template<class T> struct         Alignment    { static constexpr std::size_t value = 256; };
    static constexpr bool            gpu          = true;
    using                            I            = std::size_t;

    /**/                             CudaAllocator( int num_gpu = 0 );

    template<class T> void           deallocate   ( T *ptr, I count );
    template<class T> T*             allocate     ( I count );

    int                              num_gpu;     ///<
    I                                amount;      ///<
    std::atomic<I>                   used;        ///<
};

template<>
struct TypeInfo<CudaAllocator> {
    static std::string name() {
        return "parex::CudaAllocator";
    }
};

template<class T> void copy_memory_values( const CudaAllocator &dst_alloc, T *dst_data, const CpuAllocator  &src_alloc, const T *src_data, std::size_t len );
template<class T> void copy_memory_values( const CpuAllocator  &dst_alloc, T *dst_data, const CudaAllocator &src_alloc, const T *src_data, std::size_t len );
template<class T> void copy_memory_values( const CudaAllocator &dst_alloc, T *dst_data, const CudaAllocator &src_alloc, const T *src_data, std::size_t len );

template<class T,int n,class F> void get_memory_values( CudaAllocator &dst, CudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( CpuAllocator  &dst, CudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( CudaAllocator &dst, CpuAllocator  &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace parex

#include "CudaAllocator.tcc"

#endif // parex_GpuAllocator_H
