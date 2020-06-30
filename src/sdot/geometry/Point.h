#ifndef SDOT_POINT_H
#define SDOT_POINT_H

#include "../support/TensorOrder.h"
#include "../support/Math.h"
#include <ostream>

/**
*/
template<class TF_,int dim_>
struct Point {
    enum {                          dim            = dim_ };
    using                           TF             = TF_;

    template<class TG>              Point          ( const Point<TG,dim> &p );
    template<class TG>              Point          ( const TG *v );
    template<class TG>              Point          ( TG x, TG y, TG z );
    template<class TG>              Point          ( TG x, TG y );
    /**/                            Point          ( TF x );
    /**/                            Point          ();

    // IO
    void                            write_to_stream( std::ostream &os ) const { for( int i = 0; i < dim; ++i ) os << ( i ? "," : "" ) << data[ i ]; }
    template<class Bq> static Point read_from      ( Bq &bq );
    template<class Bq> void         write_to       ( Bq &bq ) const;

    // accessors
    const TF&                       operator[]     ( std::size_t d ) const { return data[ d ]; }
    TF&                             operator[]     ( std::size_t d ) { return data[ d ]; }
    const TF*                       begin          () const { return data; }
    const TF*                       end            () const { return data + dim; }

    bool                            operator<      ( const Point &that ) const;
    operator                        bool           () const;

    // modifiers
    Point&                          operator*=     ( TF v ) { for( int i = 0; i < dim; ++i ) data[ i ] *= v; return *this; }

    TF                              data[ dim ];
};

template<class TF,int dim>
struct TensorOrder<Point<TF,dim>> {
    enum { value = TensorOrder<TF>::value + 1 };
};

template<class TF,int dim>
TF norm_2_p2( Point<TF,dim> p ) {
    TF res = 0;
    for( int i = 0; i < dim; ++i )
        res += p[ i ] * p[ i ];
    return res;
}

template<class TF,int dim>
TF norm_2( Point<TF,dim> p ) {
    return std::sqrt( norm_2_p2( p ) );
}

template<class TF,int dim>
TF dot( Point<TF,dim> a, Point<TF,dim> b ) {
    TF res = 0;
    for( int i = 0; i < dim; ++i )
        res += a[ i ] * b[ i ];
    return res;
}

template<class TF,int dim>
Point<TF,dim> &operator+=( Point<TF,dim> &a, Point<TF,dim> b ) {
    for( int i = 0; i < dim; ++i )
        a[ i ] += b[ i ];
    return a;
}

template<class TF,int dim>
Point<TF,dim> &operator-=( Point<TF,dim> &a, Point<TF,dim> b ) {
    for( int i = 0; i < dim; ++i )
        a[ i ] -= b[ i ];
    return a;
}

template<class TF,int dim>
Point<TF,dim> &operator/=( Point<TF,dim> &a, TF b ) {
    for( int i = 0; i < dim; ++i )
        a[ i ] /= b;
    return a;
}

template<class TF,int dim>
Point<TF,dim> operator+( Point<TF,dim> a, Point<TF,dim> b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] + b[ i ];
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator+( Point<TF,dim> a, TF b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] + b;
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator-( Point<TF,dim> a, Point<TF,dim> b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] - b[ i ];
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> operator-( Point<TF,dim> a, TF b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] - b;
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> operator-( Point<TF,dim> a ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = - a[ i ];
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> operator*( TF m, Point<TF,dim> p ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = m * p[ i ];
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> operator*( Point<TF,dim> a, Point<TF,dim> b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] * b[ i ];
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> operator/( Point<TF,dim> p, TF d ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = p[ i ] / d;
    return res;
}

template<class TF,int dim>
inline bool operator==( Point<TF,dim> p, Point<TF,dim> q ) {
    for( int i = 0; i < dim; ++i )
        if ( p[ i ] != q[ i ] )
            return false;
    return true;
}

template<class TF,int dim>
inline bool operator!=( Point<TF,dim> p, Point<TF,dim> q ) {
    for( int i = 0; i < dim; ++i )
        if ( p[ i ] != q[ i ] )
            return true;
    return false;
}

template<class TF,int dim>
inline Point<TF,dim> min( Point<TF,dim> p, Point<TF,dim> q ) {
    using std::min;
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = min( p[ i ], q[ i ] );
    return res;
}

template<class TF,int dim>
inline Point<TF,dim> max( Point<TF,dim> p, Point<TF,dim> q ) {
    using std::max;
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = max( p[ i ], q[ i ] );
    return res;
}

template<class TF,int dim>
inline TF max( Point<TF,dim> p ) {
    using std::max;
    TF res = p[ 0 ];
    for( int i = 1; i < dim; ++i )
        res = max( res, p[ i ] );
    return res;
}

template<class TF,int dim>
inline TF has_nan( Point<TF,dim> p ) {
    for( int i = 0; i < dim; ++i )
        if ( std::isnan( p[ i ] ) )
            return true;
    return false;
}

template<class TF,int dim>
inline Point<TF,dim> normalized( Point<TF,dim> p, TF a = 1e-40 ) {
    return p / ( norm_2( p ) + a );
}

template<class TF>
inline Point<TF,2> cross_prod( Point<TF,2> p ) {
    return { - p[ 1 ], p[ 0 ] };
}

template<class TF>
inline Point<TF,3> cross_prod( Point<TF,3> a, Point<TF,3> b ) {
    return { a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] };
}

template<class TF,int dim>
inline Point<TF,dim> cross_prod( const Point<TF,dim> *pts ) {
    Point<TF,dim> res;
    for( int d = 0; d < dim; ++d ) {
        std::array<std::array<TF,dim-1>,dim-1> M;
        for( int r = 0; r < dim - 1; ++r )
            for( int c = 0; c < dim - 1; ++c )
                M[ r ][ c ] = pts[ r ][ c + ( c >= d ) ];
        res[ d ] = TF( d % 2 ? -1 : 1 ) * determinant( M );
    }
    return res;
}

template<class TF>
inline Point<TF,3> ortho_rand( Point<TF,3> a ) {
    for( Point<TF,3> trl : { Point<TF,3>{ 0, 1, 0 }, Point<TF,3>{ 1, 0, 0 }, Point<TF,3>{ 0, 0, 1 } } ){
        Point<TF,3> res = cross_prod( trl, a );
        TF m  = norm_2( res );
        if ( m > 1e-6 )
            return res / m;
    }
    return {};
}

template<class TF,int dim>
inline Point<TF,dim> ortho_with_normalized( Point<TF,dim> D, Point<TF,dim> N ) {
    return D - dot( D, N ) * N;
}

template<class TF>
inline Point<TF,3> transformation( const std::array<TF,9> &trans, Point<TF,3> p ) {
    return {
        trans[ 0 ] * p[ 0 ] + trans[ 1 ] * p[ 1 ] + trans[ 2 ] * p.z,
        trans[ 3 ] * p[ 0 ] + trans[ 4 ] * p[ 1 ] + trans[ 5 ] * p.z,
        trans[ 6 ] * p[ 0 ] + trans[ 7 ] * p[ 1 ] + trans[ 8 ] * p.z
    };
}

template<class TF>
inline Point<TF,2> transformation( const std::array<TF,4> &trans, Point<TF,2> p ) {
    return { trans[ 0 ] * p[ 0 ] + trans[ 1 ] * p[ 1 ], trans[ 2 ] * p[ 0 ] + trans[ 3 ] * p[ 1 ] };
}

template<class TF,int dim>
inline TF transformation( const std::array<TF,9> &/*trans*/, TF p ) {
    return p;
}

template<class TF>
inline TF transformation( const std::array<TF,4> &/*trans*/, TF val ) {
    return val;
}

#include "Point.tcc"

#endif // SDOT_POINT_H
