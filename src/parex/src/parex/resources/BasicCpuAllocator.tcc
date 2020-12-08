#include "BasicCpuAllocator.h"
#include <cstring>

#include <sys/sysinfo.h>
#include <unistd.h>

namespace parex {

template<class T> void BasicCpuAllocator::deallocate( T *ptr, I count ) {
    mem.used -= sizeof( T ) * count;
    std::free( ptr );
}

template<class T>
T *BasicCpuAllocator::allocate( I count ) {
    I size = sizeof( T ) * count;
    return reinterpret_cast<T *>( aligned_alloc( 64, size ) );
}

template<class T>
void copy_memory_values( const BasicCpuAllocator &, T *dst_data, const BasicCpuAllocator &, const T *src_data, std::size_t len ) {
    for( std::size_t i = 0; i < len; ++i )
        dst_data[ i ] = src_data[ i ];
}

template<class T,int n,class F>
void get_memory_values( BasicCpuAllocator &, BasicCpuAllocator &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f ) {
    f( ptrs );
}

} // namespace parex
