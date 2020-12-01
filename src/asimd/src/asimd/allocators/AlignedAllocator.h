#pragma once

#include <cstdlib>
#include <memory>

namespace asimd {

/** std allocator for aligned memory
 *
 * alig = alignment in bytes
 */
template<class T,std::size_t alig>
struct AlignedAllocator : std::allocator<T> {
    static constexpr size_t  alignment       = alig;
    static constexpr bool    cpu             = true;

    /**/                     AlignedAllocator() {}

    template                 <class U,std::size_t b>
    struct                   rebind          { using other = AlignedAllocator<U,b>; };

    template<class T2>       AlignedAllocator( const AlignedAllocator<T2,alig> & ) {}
    static void              deallocate      ( T *ptr, std::size_t = 0 ) { std::free( ptr ); }
    static T*                allocate        ( std::size_t count, const void * = 0 ) { return reinterpret_cast<T *>( aligned_alloc( alig, sizeof( T ) * count ) ); }
    static T                 value           ( const T *ptr ) { return *ptr; }
};

} // namespace asimd
