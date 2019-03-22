#pragma once

// #include <ostream>
#include "Assert.h"
#include <array>

/// shortcut for 1D arrays
template<class T,int dim>
using Array = std::array<T,dim>;

//template<class T,int dim>
//using Array = xt::xtensorf<T,xt::xshape<dim>>; // -> Array<...> a{{v0,v1,...}};

//// display of xt::xfixed_container (used for implementation of Array)
//template<class ET,class S,xt::layout_type L,class Tag>
//struct TensorOrder<xt::xfixed_container<ET,S,L,Tag>> { static const int res = S::size(); };

//template<class ET,class S,xt::layout_type L,class Tag,class R>
//struct ShouldBeDisplayedWriteToStream<xt::xfixed_container<ET,S,L,Tag>,R> {};
//template<class ET,class S,xt::layout_type L,class Tag>
//std::ostream &operator<<( std::ostream &os, const xt::xfixed_container<ET,S,L,Tag> &c ) {
//}


template<class T,size_t s,class U>
Array<T,s> array_of( U val ) {
    Array<T,s> res;
    res.fill( val );
    return res;
}

template<class T,size_t s,class U>
Array<T,s> array_of( const Array<U,s> &val ) {
    Array<T,s> res;
    for( std::size_t i = 0; i < s; ++i )
        res[ i ] = val[ i ];
    return res;
}

template<class T,size_t s,class U>
Array<T,s> array_of( const std::vector<U> &val ) {
    Array<T,s> res;
    if ( val.size() == 0 ) {
        res.fill( 0.0 );
    } else if ( val.size() == 1 ) {
        for( std::size_t i = 0; i < s; ++i )
            res[ i ] = val[ 0 ];
    } else if ( val.size() == s ) {
        for( std::size_t i = 0; i < s; ++i )
            res[ i ] =  val[ i ];
    } else
        ERROR( "wrong size" );
    return res;
}

template<class T,size_t s>
Array<T,s-1> tail( const Array<T,s> &val ) {
    Array<T,s-1> res;
    for( std::size_t i = 1; i < s; ++i )
        res[ i - 1 ] = val[ i ];
    return res;
}

template<class T,size_t s>
Array<T,s+1> prepend( T val, const Array<T,s> &vec ) {
    Array<T,s+1> res;
    res[ 0 ] = val;
    for( std::size_t i = 0; i < s; ++i )
        res[ i + 1 ] = vec[ i ];
    return res;
}

