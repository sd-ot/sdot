#pragma once

//#include "../Support/Stream.h"
#include <ostream>
#include <array>
#include <cmath>

namespace sdot {

/**
*/
template<class TF>
struct Point3 {
    /**/                             Point3         ( const TF *v ) : x( v[ 0 ] ), y( v[ 1 ] ), z( v[ 2 ] ) {}
    /**/                             Point3         ( TF x, TF y, TF z ) : x( x ), y( y ), z( z ) {}
    /**/                             Point3         ( TF x ) : x( x ), y( x ), z( x ) {}
    /**/                             Point3         () {}

    void                             write_to_stream( std::ostream &os ) const { os << x << " " << y << " " << z; }
    const TF&                        operator[]     ( std::size_t d ) const { return ( &x )[ d ]; }
    TF&                              operator[]     ( std::size_t d ) { return ( &x )[ d ]; }
    template<class Bq> static Point3 read_from      ( Bq &bq ) { return { TF( bq.read() ), TF( bq.read() ), TF( bq.read() ) }; }
    template<class Bq> void          write_to       ( Bq &bq ) const { bq << x << y << z; }

    TF                               x;
    TF                               y;
    TF                               z;
};

template<class TF>
inline TF norm_2_p2( Point3<TF> p ) {
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

template<class TF>
inline TF norm_2( Point3<TF> p ) {
    return std::sqrt( norm_2_p2( p ) );
}

template<class TF>
inline TF dot( Point3<TF> a, Point3<TF> b ) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<class TF>
inline Point3<TF> &operator+=( Point3<TF> &a, Point3<TF> b ) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<class TF>
inline Point3<TF> &operator-=( Point3<TF> &a, Point3<TF> b ) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<class TF>
inline Point3<TF> &operator/=( Point3<TF> &a, TF b ) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

template<class TF>
inline Point3<TF> operator+( Point3<TF> a, Point3<TF> b ) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

template<class TF>
inline Point3<TF> operator-( Point3<TF> a, Point3<TF> b ) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

template<class TF>
inline Point3<TF> operator-( Point3<TF> a ) {
    return { - a.x, - a.y, - a.z };
}

template<class TF>
inline Point3<TF> operator*( TF m, Point3<TF> p ) {
    return { m * p.x, m * p.y, m * p.z };
}

template<class TF>
inline Point3<TF> operator*( Point3<TF> a, Point3<TF> b ) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

template<class TF>
inline Point3<TF> operator/( Point3<TF> p, TF d ) {
    return { p.x / d, p.y / d, p.z / d };
}

template<class TF>
inline bool operator==( Point3<TF> p, Point3<TF> q ) {
    return p.x == q.x && p.y == q.y && p.z == q.z;
}

template<class TF>
inline bool operator!=( Point3<TF> p, Point3<TF> q ) {
    return p.x != q.x || p.y != q.y || p.z != q.z;
}

template<class TF>
inline Point3<TF> min( Point3<TF> p, Point3<TF> q ) {
    using std::min;
    return { min( p.x, q.x ), min( p.y, q.y ), min( p.z, q.z ) };
}

template<class TF>
inline Point3<TF> max( Point3<TF> p, Point3<TF> q ) {
    using std::max;
    return { max( p.x, q.x ), max( p.y, q.y ), max( p.z, q.z ) };
}

template<class TF>
inline TF max( Point3<TF> p ) {
    using std::max;
    return max( p.x, max( p.y, p.z ) );
}

template<class TF>
inline Point3<TF> normalized( Point3<TF> p, TF a = 1e-40 ) {
    auto d = norm_2( p ) + a;
    return { p.x / d, p.y / d, p.z / d };
}

template<class TF>
inline Point3<TF> cross_prod( Point3<TF> a, Point3<TF> b ) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

template<class TF>
inline Point3<TF> ortho_rand( Point3<TF> a ) {
    for( Point3<TF> trl : { Point3<TF>{ 0, 1, 0 }, Point3<TF>{ 1, 0, 0 }, Point3<TF>{ 0, 0, 1 } } ){
        Point3<TF> res = cross_prod( trl, a );
        TF m  = norm_2( res );
        if ( m > 1e-6 )
            return res / m;
    }
    return {};
}

template<class TF>
inline Point3<TF> ortho_with_normalized( Point3<TF> D, Point3<TF> N ) {
    return D - dot( D, N ) * N;
}

template<class TF>
inline Point3<TF> transformation( const std::array<TF,9> &trans, Point3<TF> p ) {
    return {
        trans[ 0 ] * p.x + trans[ 1 ] * p.y + trans[ 2 ] * p.z,
        trans[ 3 ] * p.x + trans[ 4 ] * p.y + trans[ 5 ] * p.z,
        trans[ 6 ] * p.x + trans[ 7 ] * p.y + trans[ 8 ] * p.z
    };
}

template<class TF>
inline TF transformation( const std::array<TF,9> &trans, TF p ) {
    return p;
}

} // namespace sdot
