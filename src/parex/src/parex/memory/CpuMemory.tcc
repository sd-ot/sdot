#include "CpuMemory.h"

namespace asimd {

template<class T> void CpuMemory::deallocate( T *ptr, TI count ) {
    used -= sizeof( T ) * count;
    std::free( ptr );
}

template<class T>
T *CpuMemory::allocate( TI count ) {
    TI size = sizeof( T ) * count;
    return reinterpret_cast<T *>( aligned_alloc( 64, size ) );
}

template<class T>
T CpuMemory::value( const T *ptr ) {
    return *ptr;
}

template<class T,int n,class F>
void get_memory_values( CpuMemory &, CpuMemory &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f ) {
    f( ptrs );
}

} // namespace asimd
