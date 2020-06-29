#pragma once

//#include "TypeInformation.h"
//#include "TypeConfig.h"
//#include "Assert.h"
//#include "N.h"
//#include "S.h"
#include <array>
#include <cmath>
#include "N.h"

template<class T,class U>
T ceil( T a, U m ) {
    if ( not m )
        return a;
    return ( a + m - 1 ) / m * m;
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
std::array<T,dim> solve( const std::array<std::array<T,dim>,dim> &M, const std::array<T,dim> &V ) {
    //    double r() { return double( rand() ) / RAND_MAX; }
    //    for( int i = 0; i < 40; ++i ) {
    //        std::array<std::array<double,3>,3> M{ std::array<double,3>{ r(), r(), r() }, std::array<double,3>{ r(), r(), r() }, std::array<double,3>{ r(), r(), r() } };
    //        std::array<double,3> V{ r(), r(), r() };
    //        std::array<double,3> X = solve( M, V );
    //        P( mul( M, X ), V );
    //    }

    T det = determinant( M );
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

