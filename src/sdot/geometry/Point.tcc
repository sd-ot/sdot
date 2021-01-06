#include "../support/StaticRange.h"
#include "../support/conv.h"
#include "Point.h"
#include <cmath>

namespace sdot {

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( const Point<TG,dim> &p ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = conv( p.data[ i ], parex::S<TF>() );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( const TG *v ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = TF( v[ i ] );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( TG x, TG y, TG z, TG w ) {
    static_assert( dim >= 4, "" );
    data[ 0 ] = TF( x );
    data[ 1 ] = TF( y );
    data[ 2 ] = TF( z );
    data[ 3 ] = TF( w );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( TG x, TG y, TG z ) {
    static_assert( dim >= 3, "" );
    data[ 0 ] = TF( x );
    data[ 1 ] = TF( y );
    data[ 2 ] = TF( z );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( TG x, TG y ) {
    static_assert( dim >= 2, "" );
    data[ 0 ] = TF( x );
    data[ 1 ] = TF( y );
}

template<class TF,int dim>
Point<TF,dim>::Point( TF x ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = x;
}

template<class TF,int dim>
Point<TF,dim>::Point() {
}

// IO
template<class TF,int dim> template<class Bq>
Point<TF,dim> Point<TF,dim>::read_from( Bq &bq )  {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res.data[ i ] = bq.read();
    return res;
}

template<class TF,int dim> template<class Bq>
void Point<TF,dim>::write_to( Bq &bq ) const {
    for( int i = 0; i < dim; ++i )
        bq << data[ i ];
}

template<class TF,int dim>
bool Point<TF,dim>::operator<( const Point &that ) const {
    for( int i = 0; i < dim; ++i )
        if ( data[ i ] != that.data[ i ] )
            return data[ i ] < that.data[ i ];
    return false;
}

template<class TF,int dim>
Point<TF,dim>::operator bool() const {
    for( int i = 0; i < dim; ++i )
        if ( data[ i ] )
            return true;
    return false;
}


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
Point<TF,dim> &operator*=( Point<TF,dim> &a, TF b ) {
    for( int i = 0; i < dim; ++i )
        a[ i ] *= b;
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
Point<TF,dim> operator-( Point<TF,dim> a, TF b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] - b;
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator-( Point<TF,dim> a ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = - a[ i ];
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator*( TF m, Point<TF,dim> p ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = m * p[ i ];
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator*( Point<TF,dim> a, Point<TF,dim> b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] * b[ i ];
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator/( Point<TF,dim> p, TF d ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = p[ i ] / d;
    return res;
}

template<class TF,int dim>
Point<TF,dim> operator/( Point<TF,dim> a, Point<TF,dim> b ) {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = a[ i ] / b[ i ];
    return res;
}

template<class TF,int dim>
bool operator==( Point<TF,dim> p, Point<TF,dim> q ) {
    for( int i = 0; i < dim; ++i )
        if ( p[ i ] != q[ i ] )
            return false;
    return true;
}

template<class TF,int dim>
bool operator!=( Point<TF,dim> p, Point<TF,dim> q ) {
    for( int i = 0; i < dim; ++i )
        if ( p[ i ] != q[ i ] )
            return true;
    return false;
}

template<class TF,int dim>
Point<TF,dim> min( Point<TF,dim> p, Point<TF,dim> q ) {
    using std::min;
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = min( p[ i ], q[ i ] );
    return res;
}

template<class TF,int dim>
Point<TF,dim> max( Point<TF,dim> p, Point<TF,dim> q ) {
    using std::max;
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res[ i ] = max( p[ i ], q[ i ] );
    return res;
}

template<class TF,int dim>
TF max( Point<TF,dim> p ) {
    using std::max;
    TF res = p[ 0 ];
    for( int i = 1; i < dim; ++i )
        res = max( res, p[ i ] );
    return res;
}

template<class TF,int dim>
TF has_nan( Point<TF,dim> p ) {
    for( int i = 0; i < dim; ++i )
        if ( std::isnan( p[ i ] ) )
            return true;
    return false;
}

template<class TF,int dim>
Point<TF,dim> normalized( Point<TF,dim> p, TF a ) {
    return p / ( norm_2( p ) + a );
}

template<class TF>
Point<TF,2> cross_prod( Point<TF,2> p ) {
    return { - p[ 1 ], p[ 0 ] };
}

template<class TF>
Point<TF,3> cross_prod( Point<TF,3> a, Point<TF,3> b ) {
    return { a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] };
}

template<class TF>
Point<TF,1> cross_prod( const Point<TF,1> * ) {
    return TF( 1 );
}

template<class TF>
Point<TF,2> cross_prod( const Point<TF,2> *pts ) {
    return cross_prod( pts[ 0 ] );
}

template<class TF>
Point<TF,3> cross_prod( const Point<TF,3> *pts ) {
    return cross_prod( pts[ 0 ], pts[ 1 ] );
}

template<class TF,int dim>
Point<TF,dim> cross_prod( const Point<TF,dim> *pts ) {
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
Point<TF,3> ortho_rand( Point<TF,3> a ) {
    for( Point<TF,3> trl : { Point<TF,3>{ 0, 1, 0 }, Point<TF,3>{ 1, 0, 0 }, Point<TF,3>{ 0, 0, 1 } } ){
        Point<TF,3> res = cross_prod( trl, a );
        TF m  = norm_2( res );
        if ( m > 1e-6 )
            return res / m;
    }
    return {};
}

template<class TF,int dim>
Point<TF,dim> ortho_norm( Point<TF,dim> D, Point<TF,dim> N ) {
    return D - dot( D, N ) * N;
}

template<class TF,int dim>
std::vector<Point<TF,dim>> base_from( const std::vector<Point<TF,dim>> &dirs ) {
    std::vector<Point<TF,dim>> res;
    StaticRange<1,dim+1>::for_each_cont( [&]( auto d ) {
        // try to add a new dir
        for( const Point<TF,dim> &pd : dirs ) {
            std::array<std::array<TF,d.value>,d.value> M;
            for( std::size_t r = 0; r < d.value; ++r )
                for( std::size_t c = 0; c < d.value; ++c )
                    M[ r ][ c ] = dot(
                        r + 1 == d.value ? pd : res[ r ],
                        c + 1 == d.value ? pd : res[ c ]
                    );
            if ( determinant( M ) ) {
                res.push_back( pd );
                return true;
            }
        }

        return false;
    } );

    return res;
}


template<class TF,int dim>
int rank_pts( const std::vector<Point<TF,dim>> &pts ) {
    int res = 0;
    std::vector<Point<TF,dim>> dirs;
    StaticRange<1,dim+1>::for_each_cont( [&]( auto d ) {
        // try to add a new dir
        for( std::size_t i = 1; i < pts.size(); ++i ) {
            Point<TF,dim> pd = pts[ i ] - pts[ 0 ];
            std::array<std::array<TF,d.value>,d.value> M;
            for( std::size_t r = 0; r < d.value; ++r )
                for( std::size_t c = 0; c < d.value; ++c )
                    M[ r ][ c ] = dot(
                        r + 1 == d.value ? pd : dirs[ r ],
                        c + 1 == d.value ? pd : dirs[ c ]
                    );
            if ( determinant( M ) ) {
                dirs.push_back( pd );
                res = d.value;
                return true;
            }
        }

        return false;
    } );

    return res;
}

}
