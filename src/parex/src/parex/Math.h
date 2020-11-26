#pragma once

#include <array>
#include <cmath>
#include "N.h"

template<class T,class U>
T ceil( T a, U m ) {
    if ( not m )
        return a;
    return ( a + m - 1 ) / m * m;
}

template<class T>
T factorial( T val ) {
    return val ? val * factorial( val - T( 1 ) ) : T( 1 );
}

template<class T,class U>
T gcd( T a, U b ) {
    if ( b == 1 )
        return 1;

    T old;
    while ( b ) {
        old = b;
        b = a % b;
        a = old;
    }
    return a;
}

template<class T,class U>
T lcm( T a, U b ) {
    return a * b / gcd( a, b );
}

template<class T>
T determinant( const T *v, N<1> ) {
    return v[ 0 ];
}

template<class T,int dim>
T determinant( const T *v, N<dim> ) {
    T res = 0;
    for( int i = 0; i < dim; ++i ) {
        std::array<std::array<T,dim-1>,dim-1> n;
        for( int r = 0; r < dim - 1; ++r )
            for( int c = 0; c < dim - 1; ++c )
                n[ r ][ c ] = v[ ( r + ( r >= i ) ) * dim + c + 1 ];
        res += T( i % 2 ? -1 : 1 ) * v[ i * dim ] * determinant( &n[ 0 ][ 0 ], N<dim-1>() );
    }
    return res;
}

template<class T,std::size_t dim>
T determinant( const std::array<std::array<T,dim>,dim> &v ) {
    return determinant( &v[ 0 ][ 0 ], N<dim>() );
}

template<class T,std::size_t dim>
std::array<T,dim> mul( const std::array<std::array<T,dim>,dim> &M, const std::array<T,dim> &V ) {
    std::array<T,dim> res;
    for( int r = 0; r < dim ; ++r ) {
        res[ r ] = 0;
        for( int c = 0; c < dim; ++c )
            res[ r ] += M[ r ][ c ] * V[ c ];
    }
    return res;
}

template<class T,std::size_t dim>
std::array<T,dim> solve( const std::array<std::array<T,dim>,dim> &M, const std::array<T,dim> &V, bool *ok = nullptr ) {
    T det = determinant( M );
    if ( det == 0 ) {
        if ( ok )
            *ok = false;
        return {};
    }

    std::array<T,dim> res;
    for( int i = 0; i < dim; ++i ) {
        std::array<std::array<T,dim>,dim> n;
        for( int r = 0; r < dim ; ++r )
            for( int c = 0; c < dim; ++c )
                n[ r ][ c ] = c == i ? V[ r ] : M[ r ][ c ];
        res[ i ] = T( i % 2 ? 1 : 1 ) * determinant( n ) / det;
    }
    return res;
}

template<class Pt>
bool colinear( const Pt &a, const Pt &b ) {
    auto d = dot( a, b );
    return d * d == norm_2_p2( a ) * norm_2_p2( b );
}

template<class V>
auto mean( const V &v ) -> typename std::decay<decltype( v[ 0 ] )>::type {
    using T = typename std::decay<decltype( v[ 0 ] )>::type;
    T res = v[ 0 ];
    for( std::size_t i = 1; i < v.size(); ++i )
        res += v[ i ];
    return res / T( v.size() );
}

template<class V,class F>
auto mean( const V &v, const F &f ) -> typename std::decay<decltype( f( v[ 0 ] ) )>::type {
    using T = typename std::decay<decltype( f( v[ 0 ] ) )>::type;
    T res = f( v[ 0 ] );
    for( std::size_t i = 1; i < v.size(); ++i )
        res += f( v[ i ] );
    return res / T( v.size() );
}
