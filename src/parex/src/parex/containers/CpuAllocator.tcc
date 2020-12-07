#include "CpuAllocator.h"
#include <cstring>

namespace parex {

template<class T> void CpuAllocator::deallocate( T *ptr, I count ) {
    used -= sizeof( T ) * count;
    std::free( ptr );
}

template<class T>
T *CpuAllocator::allocate( I count ) {
    I size = sizeof( T ) * count;
    return reinterpret_cast<T *>( aligned_alloc( 64, size ) );
}

template<class T>
void copy_memory_values( const CpuAllocator &, T *dst_data, const CpuAllocator &, const T *src_data, std::size_t len ) {
    for( std::size_t i = 0; i < len; ++i )
        dst_data[ i ] = src_data[ i ];
}

template<class T,int n,class F>
void get_memory_values( CpuAllocator &, CpuAllocator &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f ) {
    f( ptrs );
}

} // namespace parex
