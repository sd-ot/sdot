#include "ZGridDiracSetStd.h"
#include "../support/Math.h"
#include <new>

namespace sdot {

template<class ItemPerDirac>
ZGridDiracSetStd<ItemPerDirac>::~ZGridDiracSetStd() {
    ST *data = ids();
    for( ST i = _size; i--; )
        data[ i ].~ST();

    for( ST d = DIM + 1; d--; ) {
        TF *data = coords( d );
        for( ST i = _size; i--; )
            data[ i ].~TF();
    }
}

template<class ItemPerDirac>
ZGridDiracSetStd<ItemPerDirac> *ZGridDiracSetStd<ItemPerDirac>::New( ST size ) {
    ST rese = ceil( size, alf ), rb = ceil( sizeof( ZGridDiracSetStd ), alb ) + ( DIM + 1 ) * rese * sizeof( TF ) + rese + sizeof( ST );

    ZGridDiracSetStd *res = new ( aligned_alloc( alf * sizeof( TF ), rb ) ) ZGridDiracSetStd;
    res->_rese = rese;
    res->_size = size;

    for( ST d = 0; d < DIM + 1; ++d ) {
        TF *data = res->coords( d );
        for( ST i = 0; i < size; ++i )
            new ( data + i ) TF;
    }

    ST *data = res->ids();
    for( ST i = 0; i < size; ++i )
        new ( data + i ) ST;

    return res;
}

template<class ItemPerDirac>
void ZGridDiracSetStd<ItemPerDirac>::get_base_data( TF **coords, TF *&weights, ST *&ids ) {
    for( ST dim = 0; dim < DIM; ++dim )
        coords[ dim ] = this->coords( dim );
    weights = this->coords( DIM );
    ids = this->ids();
}

template<class ItemPerDirac>
ST ZGridDiracSetStd<ItemPerDirac>::size() {
    return _size;
}

template<class ItemPerDirac>
TF* ZGridDiracSetStd<ItemPerDirac>::coords( int dim ) {
    char *p = reinterpret_cast<char *>( this ) + ceil( sizeof( ZGridDiracSetStd ), alb );
    return reinterpret_cast<TF *>( p ) + dim * _rese;
}

template<class ItemPerDirac>
ST* ZGridDiracSetStd<ItemPerDirac>::ids() {
    return reinterpret_cast<ST *>( coords( DIM + 1 ) );
}

}
