#include "../utility/TODO.h"
#include "GpuAllocator.h"

namespace parex {

GpuAllocator::GpuAllocator( int num_gpu ) : num_gpu( num_gpu ), used( 0 ) {
}

template<class T> void GpuAllocator::deallocate( T *ptr, I count ) {
    I size = sizeof( T ) * count;
    used -= size;

    cudaSetDevice( num_gpu );
    cudaFree( ptr );
}

template<class T>
T *GpuAllocator::allocate( I count ) {
    I size = sizeof( T ) * count;
    used += size;

    T *res;
    cudaSetDevice( num_gpu );
    cudaMalloc( &res, size );
    return res;
}

template<class T>
T GpuAllocator::value( const T *ptr ) {
    T res;
    cudaSetDevice( num_gpu );
    cudaMemcpy( &res, ptr, sizeof( T ), cudaMemcpyDeviceToHost );
    return res;
}

template<class T,int n,class F>
void get_memory_values( GpuAllocator &dst, GpuAllocator &src, std::array<std::pair<const T *,std::size_t>,n> src_ptrs, F &&f ) {
    if ( dst.num_gpu == src.num_gpu ) {
        f( src_ptrs );
        return;
    }

    // != gpus
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );
        cudaMemcpyPeer( const_cast<T *>( dst_ptrs[ i ].first ), dst.num_gpu, src_ptrs[ i ].first, src.num_gpu, sizeof( T ) * src_ptrs[ i ].second, cudaMemcpyHostToDevice );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

template<class T,int n,class F>
void get_memory_values( CpuAllocator &dst, GpuAllocator &src, std::array<std::pair<const T *,std::size_t>,n> src_ptrs , F &&f ) {
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );

        cudaSetDevice( src.num_gpu );
        cudaMemcpy( const_cast<T *>( dst_ptrs[ i ].first ), src_ptrs[ i ].first, sizeof( T ) * src_ptrs[ i ].second, cudaMemcpyDeviceToHost );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

template<class T,int n,class F>
void get_memory_values( GpuAllocator &dst, CpuAllocator &/*src*/, std::array<std::pair<const T *,std::size_t>,n> src_ptrs, F &&f ) {
    std::array<std::pair<const T *,std::size_t>,n> dst_ptrs;
    for( int i = 0; i < n; ++i ) {
        dst_ptrs[ i ].first = dst.allocate<T>( src_ptrs[ i ].second );

        cudaSetDevice( dst.num_gpu );
        cudaMemcpy( const_cast<T *>( dst_ptrs[ i ].first ), src_ptrs[ i ].first, sizeof( T ) * src_ptrs[ i ].second, cudaMemcpyHostToDevice );

        dst_ptrs[ i ].second = src_ptrs[ i ].second;
    }

    f( dst_ptrs );

    for( int i = 0; i < n; ++i )
        dst.deallocate( dst_ptrs[ i ].first, dst_ptrs[ i ].second );
}

} // namespace parex
