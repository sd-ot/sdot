#ifndef parex_GpuAllocator_H
#define parex_GpuAllocator_H

#include "BasicCpuAllocator.h"
#include "CudaMemory.h"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

//// nsmake lib_name cudart

namespace parex {

/** std allocator for aligned memory */
class BasicCudaAllocator {
public:
    template<class T> struct Alignment         { static constexpr std::size_t value = 256; };
    static constexpr bool    gpu               = true;
    using                    I                 = std::size_t;

    /**/                     BasicCudaAllocator( int num_gpu );
    template<class T> void   deallocate        ( T *ptr, I count );
    template<class T> T*     allocate          ( I count );
    CudaMemory*              memory            () { return &mem; }

    CudaMemory               mem;              ///<
};

template<>
struct TypeInfo<BasicCudaAllocator> {
    static std::string name() {
        return "parex::BasicCudaAllocator";
    }
};

template<class T> void copy_memory_values( const BasicCudaAllocator &dst_alloc, T *dst_data, const BasicCpuAllocator  &src_alloc, const T *src_data, std::size_t len );
template<class T> void copy_memory_values( const BasicCpuAllocator  &dst_alloc, T *dst_data, const BasicCudaAllocator &src_alloc, const T *src_data, std::size_t len );
template<class T> void copy_memory_values( const BasicCudaAllocator &dst_alloc, T *dst_data, const BasicCudaAllocator &src_alloc, const T *src_data, std::size_t len );

template<class T,int n,class F> void get_memory_values( BasicCudaAllocator &dst, BasicCudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( BasicCpuAllocator  &dst, BasicCudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );
template<class T,int n,class F> void get_memory_values( BasicCudaAllocator &dst, BasicCpuAllocator  &src, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace parex

#include "BasicCudaAllocator.tcc"

#endif // parex_GpuAllocator_H
