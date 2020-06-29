#include "AlignedAllocator.h"
#include "StructOfArrays.h"
#include "simd/SimdVec.h"

template<class Attributes,class Arch,class TI>
StructOfArrays<Attributes,Arch,TI>::StructOfArrays( const std::vector<TI> &vector_sizes, TI rese ) : rese( rese ), size( 0 ) {
    const TI *v = vector_sizes.data();
    data.init( v );

    for_each_ptr( [&]( auto *&t, auto s ) {
        using T = typename decltype( s )::T;
        t = AlignedAllocator::allocate<T,SimdAlig<T,Arch>::value>( rese );
    } );
}

template<class Attributes,class Arch,class TI>
StructOfArrays<Attributes,Arch,TI>::StructOfArrays( StructOfArrays &&that ) : data( std::move( that.data ) ) {
    rese = that.rese;
    size = that.size;
    that.rese = 0;
    that.size = 0;
}

template<class Attributes,class Arch,class TI>
StructOfArrays<Attributes,Arch,TI>::~StructOfArrays() {
    for_each_ptr( [&]( auto *&t, auto ) {
        AlignedAllocator::free( t, rese );
    } );
}

template<class Attributes,class Arch,class TI> template<class F>
void StructOfArrays<Attributes,Arch,TI>::for_each_ptr( const F &f ) {
    data.for_each_ptr( f );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::clear() {
    size = 0;
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::reserve( TI new_rese, TI old_size ) {
    // nothing to do ?
    TI old_rese = rese;
    if ( rese >= new_rese )
        return;

    // find the reservation size
    rese += rese == 0;
    while ( rese < new_rese )
        rese *= 2;

    // realloc
    for_each_ptr( [&]( auto *&t, auto s ) {
        using T = typename decltype( s )::T;
        AlignedAllocator::reallocate<T,SimdAlig<T,Arch>::value>( t, old_size, old_rese, rese );
    } );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::reserve( TI new_rese ) {
    reserve( new_rese, size );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::write_to_stream( std::ostream &os ) const {
    for( TI i = 0; i < size; ++i ) {
        if ( i )
            os << "\n";
        TI cpt = 0;
        data.write_to_stream( os, i, cpt );
    }
}
