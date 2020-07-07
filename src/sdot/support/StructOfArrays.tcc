#include "AlignedAllocator.h"
#include "StructOfArrays.h"
#include "simd/SimdVec.h"

template<class Attributes,class Arch,class TI>
StructOfArrays<Attributes,Arch,TI>::StructOfArrays( const std::vector<TI> &vector_sizes, TI rese ) : rese( rese ), size( 0 ) {
    const TI *v = vector_sizes.data();
    data.init( v );

    for_each_ptr( [&]( auto *&t, auto s ) {
        using T = typename decltype( s )::T;
        if ( rese ) {
            t = AlignedAllocator<T,Arch>::allocate( rese );
            for( TI i = 0; i < rese; ++i )
                new ( t + i ) T;
        }
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
    for_each_ptr( [&]( auto *t, auto s ) {
        using T = typename decltype( s )::T;
        if ( rese ) {
            if ( ! std::is_trivially_destructible<T>::value )
                while ( rese-- )
                    t[ rese ].~T();
            AlignedAllocator<T,Arch>::deallocate( t, rese );
        }
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
    if ( rese >= new_rese )
        return;

    // find the reservation size
    TI old_rese = rese;
    rese += rese == 0;
    while ( rese < new_rese )
        rese *= 2;

    // realloc
    for_each_ptr( [&]( auto *&t, auto s ) {
        using T = typename decltype( s )::T;

        T *old_t = t;
        t = AlignedAllocator<T,Arch>::allocate( rese );

        for( TI i = 0; i < old_size; ++i )
            new ( t + i ) T( std::move( old_t[ i ] ) );
        for( TI i = old_size; i < new_rese; ++i )
            new ( t + i ) T;

        if ( old_rese ) {
            if ( ! std::is_trivially_destructible<T>::value )
                while ( old_rese-- )
                    old_t[ old_rese ].~T();
            std::free( old_t );
        }
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
