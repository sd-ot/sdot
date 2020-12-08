#include "BasicCudaAllocator.h"
#include "../utility/TODO.h"

#if __has_include(<cuda_runtime.h>)

namespace parex {

template<class T> void BasicCudaAllocator::deallocate( T *ptr, I count ) {
    I size = sizeof( T ) * count;
    mem.used -= size;

    cudaSetDevice( mem.num_gpu );
    cudaFree( ptr );
}

template<class T>
T *BasicCudaAllocator::allocate( I count ) {
    I size = sizeof( T ) * count;
    mem.used += size;

    T *res;
    cudaSetDevice( mem.num_gpu );
    cudaMalloc( &res, size );
    return res;
}

template<class T>
void copy_memory_values( const BasicCudaAllocator &dst_alloc, T *dst_data, const BasicCpuAllocator &, const T *src_data, std::size_t len ) {
    cudaSetDevice( dst_alloc.mem.num_gpu );
    cudaMemcpy( dst_data, src_data, sizeof( T ) * len, cudaMemcpyHostToDevice );
}

template<class T>
void copy_memory_values( const BasicCpuAllocator  &, T *dst_data, const BasicCudaAllocator &src_alloc, const T *src_data, std::size_t len ) {
    cudaSetDevice( src_alloc.mem.num_gpu );
    cudaMemcpy( dst_data, src_data, sizeof( T ) * len, cudaMemcpyDeviceToHost );
}

template<class T>
void copy_memory_values( const BasicCudaAllocator &dst_alloc, T *dst_data, const BasicCudaAllocator &src_alloc, const T *src_data, std::size_t len ) {
    cudaMemcpyPeer( dst_data, dst_alloc.mem.num_gpu, src_data, src_alloc.mem.num_gpu, sizeof( T ) * len );
}

template<class T,int n,class F>
void get_memory_values( BasicCudaAllocator &dst, BasicCudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> src_ptrs, F &&f ) {
    if ( dst.mem.num_gpu == src.mem.num_gpu ) {
        f( src_ptrs );
        return;
    }

    // != gpus
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );
        cudaMemcpyPeer( const_cast<T *>( dst_ptrs[ i ].first ), dst.mem.num_gpu, src_ptrs[ i ].first, src.mem.num_gpu, sizeof( T ) * src_ptrs[ i ].second );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

template<class T,int n,class F>
void get_memory_values( BasicCpuAllocator &dst, BasicCudaAllocator &src, std::array<std::pair<const T *,std::size_t>,n> src_ptrs , F &&f ) {
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );

        cudaSetDevice( src.mem.num_gpu );
        cudaMemcpy( const_cast<T *>( dst_ptrs[ i ].first ), src_ptrs[ i ].first, sizeof( T ) * src_ptrs[ i ].second, cudaMemcpyDeviceToHost );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

template<class T,int n,class F>
void get_memory_values( BasicCudaAllocator &dst, BasicCpuAllocator &/*src*/, std::array<std::pair<const T *,std::size_t>,n> src_ptrs, F &&f ) {
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );

        cudaSetDevice( dst.mem.num_gpu );
        cudaMemcpy( const_cast<T *>( dst_ptrs[ i ].first ), src_ptrs[ i ].first, sizeof( T ) * src_ptrs[ i ].second, cudaMemcpyHostToDevice );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

} // namespace parex

#endif
