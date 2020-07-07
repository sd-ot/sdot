#pragma once

#include "simd/SimdVec.h"
//#include <type_traits>
#include <utility>
#include <cstdlib>

template<class T,class Arch>
struct AlignedAllocator {
    using                    value_type      = T                ;
    using                    pointer         = value_type*      ;
    using                    const_pointer   = const value_type*;
    using                    reference       = value_type&      ;
    using                    const_reference = const value_type&;
    using                    size_type       = std::size_t      ;
    using                    difference_type = std::ptrdiff_t   ;
    template<class U> struct rebind          { using other = AlignedAllocator<U,Arch>; };

    /**/                     AlignedAllocator() {}
    /**/                    ~AlignedAllocator() {}

    template<class T2>       AlignedAllocator( const AlignedAllocator<T2,Arch> & ) {}

    template<class T2> bool  operator==      ( const AlignedAllocator<T2,Arch> & ) const { return true; }
    template<class T2> bool  operator!=      ( const AlignedAllocator<T2,Arch> & ) const { return false; }
    static void              deallocate      ( pointer ptr, size_type ) { std::free( ptr ); }
    static void              construct       ( pointer ptr, const T &t ) { new( ptr ) T( t ); }
    static pointer           allocate        ( size_type count, const void* = 0 ) { return reinterpret_cast<pointer>( aligned_alloc( alignof( T ) * SimdSize<T,Arch>::value, sizeof( T ) * count ) ); }
    static size_type         max_size        () { return 0xffffffffUL / sizeof( T ); }
    static pointer           address         ( reference ref ) { return &ref; }
    static const_pointer     address         ( const_reference ref ) { return &ref; }
    static void              destroy         ( pointer ptr ) { ptr->~T(); }
};
