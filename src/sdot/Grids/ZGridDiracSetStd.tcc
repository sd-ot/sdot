#include "ZGridDiracSetStd.h"
#include "../support/Math.h"
#include <new>

namespace sdot {

template<class Arch,class T,class S,int dim,class ItemPerDirac>
ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::~ZGridDiracSetStd() {
    S *data = ids();
    for( S i = _size; i--; )
        data[ i ].~S();

    for( S d = dim + 1; d--; ) {
        T *data = coords( d );
        for( S i = _size; i--; )
            data[ i ].~T();
    }
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac> *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::New( S size ) {
    S rese = ceil( size, alf ), rb = ceil( sizeof( ZGridDiracSetStd ), alb ) + ( dim + 1 ) * rese * sizeof( T ) + rese + sizeof( S );

    ZGridDiracSetStd *res = new ( aligned_alloc( alf * sizeof( T ), rb ) ) ZGridDiracSetStd;
    res->_rese = rese;
    res->_size = size;

    for( S d = 0; d < dim + 1; ++d ) {
        T *data = res->coords( d );
        for( S i = 0; i < size; ++i )
            new ( data + i ) T;
    }

    S *data = res->ids();
    for( S i = 0; i < size; ++i )
        new ( data + i ) S;

    return res;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
void ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::get_base_data( T **coords, T *&weights, S *&ids ) {
    for( S d = 0; d < dim; ++d )
        coords[ d ] = this->coords( d );
    weights = this->coords( dim );
    ids = this->ids();
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
S ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::size() {
    return _size;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
T* ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::coords( int d ) {
    char *p = reinterpret_cast<char *>( this ) + ceil( sizeof( ZGridDiracSetStd ), alb );
    return reinterpret_cast<T *>( p ) + d * _rese;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
S* ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::ids() {
    return reinterpret_cast<S *>( coords( dim + 1 ) );
}

}
