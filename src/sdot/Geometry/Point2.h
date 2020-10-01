
#pragma once

#include <ostream>
#include <array>
#include <cmath>

namespace sdot {

/**
*/
template<class TF>
struct Point2 {
    /**/                             Point2         ( const TF *v ) : x( v[ 0 ] ), y( v[ 1 ] ) {}
    /**/                             Point2         ( TF x, TF y ) : x( x ), y( y ) {}
    /**/                             Point2         ( TF x ) : x( x ), y( x ) {}
    /**/                             Point2         () {}

    void                             write_to_stream( std::ostream &os ) const { os << x << " " << y; }
    const TF&                        operator[]     ( std::size_t d ) const { return ( &x )[ d ]; }
    TF&                              operator[]     ( std::size_t d ) { return ( &x )[ d ]; }
    template<class Bq> static Point2 read_from      ( Bq &bq ) { return { TF( bq.read() ), TF( bq.read() ) }; }
    template<class Bq> void          write_to       ( Bq &bq ) const { bq << x << y; }

    TF                               x;
    TF                               y;
};

template<class TF>
inline TF norm_2_p2( Point2<TF> p ) {
    return p.x * p.x + p.y * p.y;
}


template<class TF>
inline TF norm_2( Point2<TF> p ) {
    using std::sqrt;
    return sqrt( norm_2_p2( p ) );
}

template<class TF>
inline TF dot( Point2<TF> a, Point2<TF> b ) {
    return a.x * b.x + a.y * b.y;
}

template<class TF>
inline Point2<TF> &operator+=( Point2<TF> &a, Point2<TF> b ) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<class TF>
inline Point2<TF> &operator-=( Point2<TF> &a, Point2<TF> b ) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<class TF>
inline Point2<TF> &operator*=( Point2<TF> &a, TF b ) {
    a.x *= b;
    a.y *= b;
    return a;
}

template<class TF>
inline Point2<TF> operator+( Point2<TF> a, Point2<TF> b ) {
    return { a.x + b.x, a.y + b.y };
}

template<class TF>
inline Point2<TF> operator+( Point2<TF> a, TF b ) {
    return { a.x + b, a.y + b };
}

template<class TF>
inline Point2<TF> operator-( Point2<TF> a, Point2<TF> b ) {
    return { a.x - b.x, a.y - b.y };
}

template<class TF>
inline Point2<TF> operator-( Point2<TF> a, TF b ) {
    return { a.x - b, a.y - b };
}

template<class TF>
inline Point2<TF> operator-( Point2<TF> a ) {
    return { - a.x, - a.y };
}

template<class TF>
inline Point2<TF> operator*( TF m, Point2<TF> p ) {
    return { m * p.x, m * p.y };
}

template<class TF>
inline Point2<TF> operator*( Point2<TF> a, Point2<TF> b ) {
    return { a.x * b.x, a.y * b.y };
}

template<class TF>
inline Point2<TF> operator/( Point2<TF> p, TF d ) {
    return { p.x / d, p.y / d };
}

template<class TF>
inline bool operator==( Point2<TF> p, Point2<TF> q ) {
    return p.x == q.x && p.y == q.y;
}

template<class TF>
inline bool operator!=( Point2<TF> p, Point2<TF> q ) {
    return p.x != q.x || p.y != q.y;
}

template<class TF>
inline Point2<TF> min( Point2<TF> p, Point2<TF> q ) {
    using std::min;
    return { min( p.x, q.x ), min( p.y, q.y ) };
}

template<class TF>
inline Point2<TF> max( Point2<TF> p, Point2<TF> q ) {
    using std::max;
    return { max( p.x, q.x ), max( p.y, q.y ) };
}

template<class TF>
inline TF max( Point2<TF> p ) {
    using std::max;
    return max( p.x, p.y );
}

template<class TF>
inline Point2<TF> normalized( Point2<TF> p, TF a = 1e-40 ) {
    TF d = norm_2( p ) + a;
    return { p.x / d, p.y / d };
}

template<class TF>
inline Point2<TF> rot90( Point2<TF> p ) {
    return { - p.y, p.x };
}

template<class TF>
inline Point2<TF> transformation( const std::array<TF,4> &trans, Point2<TF> p ) {
    return { trans[ 0 ] * p.x + trans[ 1 ] * p.y, trans[ 2 ] * p.x + trans[ 3 ] * p.y };
}

template<class TF>
inline TF transformation( const std::array<TF,4> &trans, TF val ) {
    return val;
}

} // namespace sdot
