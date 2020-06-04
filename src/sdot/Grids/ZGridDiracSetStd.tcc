#include "ZGridDiracSetStd.h"
#include "../support/Math.h"
#include <new>

namespace sdot {

template<class Arch,class T,class S,int dim,class ItemPerDirac>
ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac> *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::New( BumpPointerPool &pool, S size ) {
    S rese = ceil( size, alf ), rb = ceil( sizeof( ZGridDiracSetStd ), alb ) + ( dim + 1 ) * rese * sizeof( T ) + rese * sizeof( S );
    ZGridDiracSetStd *res = new ( pool.allocate( rb, alb ) ) ZGridDiracSetStd;
    res->_rese = rese;
    res->_size = 0;

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
S ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::nb_diracs_for_mem( std::size_t mem ) {
    return mem / ( ( dim + 1 ) * sizeof( T ) + sizeof( ItemPerDirac ) );
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
void ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::write_to_stream( std::ostream &os, const std::string &sp ) const {
    for( ST i = 0; i < _size; ++i ) {
        os << sp;
        for( ST d = 0; d < dim; ++d )
            os << ( d ? " " : "" ) << coords( d )[ i ];
        os << "\n";
    }
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
void ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::get_base_data( T **coords, T *&weights, S *&ids ) {
    for( S d = 0; d < dim; ++d )
        coords[ d ] = this->coords( d );
    weights = this->weights();
    ids = this->ids();
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
S ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::size() {
    return _size;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
void ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::add_dirac( const T *c, T w, S id ) {
    for( ST d = 0; d < dim; ++d )
        coords( d )[ _size ] = c[ d ];
    weights()[ _size ] = w;
    ids()[ _size ] = id;
    ++_size;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
const T *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::coords( int d ) const {
    const char *p = reinterpret_cast<const char *>( this ) + ceil( sizeof( ZGridDiracSetStd ), alb );
    return reinterpret_cast<const T *>( p ) + d * _rese;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
T *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::coords( int d ) {
    char *p = reinterpret_cast<char *>( this ) + ceil( sizeof( ZGridDiracSetStd ), alb );
    return reinterpret_cast<T *>( p ) + d * _rese;
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
const T *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::weights() const {
    return coords( dim );
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
T* ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::weights() {
    return coords( dim );
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
const S *ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::ids() const {
    return reinterpret_cast<const S *>( coords( dim + 1 ) );
}

template<class Arch,class T,class S,int dim,class ItemPerDirac>
S* ZGridDiracSetStd<Arch,T,S,dim,ItemPerDirac>::ids() {
    return reinterpret_cast<S *>( coords( dim + 1 ) );
}

}
