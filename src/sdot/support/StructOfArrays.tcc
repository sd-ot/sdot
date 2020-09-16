#include "StructOfArrays.h"

namespace sdot {

template<class Attributes,class Arch,class TI>
StructOfArrays<Attributes,Arch,TI>::StructOfArrays( const std::vector<TI> &vector_sizes, TI rese ) : _rese( rese ), _size( 0 ) {
    const TI *v = vector_sizes.data();
    _data.set_vector_sizes( v );

    for_each_vec( [&]( auto &vec ) {
        vec.resize( rese );
    } );
}

template<class Attributes,class Arch,class TI> template<class F>
void StructOfArrays<Attributes,Arch,TI>::for_each_vec( const F &f ) {
    _data.for_each_vec( f );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::clear() {
    _size = 0;
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::reserve( TI new_rese, TI /*old_size*/ ) {
    // nothing to do ?
    if ( _rese >= new_rese )
        return;

    // find the reservation size
    // TI old_rese = _rese;
    _rese += _rese == 0;
    while ( _rese < new_rese )
        _rese *= 2;

    // realloc
    for_each_vec( [&]( auto &vec ) {
        vec.resize( _rese );
    } );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::reserve( TI new_rese ) {
    reserve( new_rese, _size );
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::write_to_stream( std::ostream &os ) const {
    for( TI i = 0; i < _size; ++i ) {
        if ( i )
            os << "\n";
        TI cpt = 0;
        _data.write_to_stream( os, i, cpt );
    }
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::resize_wo_check( TI new_size ) {
    _size = new_size;
}

template<class Attributes,class Arch,class TI>
void StructOfArrays<Attributes,Arch,TI>::resize( TI new_size ) {
    reserve( new_size );
    _size = new_size;
}

template<class Attributes,class Arch,class TI>
TI StructOfArrays<Attributes,Arch,TI>::size() const {
    return _size;
}

} // namespace sdot
