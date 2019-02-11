#pragma once

#include <type_traits>
#include <cmath>

#include "TypePromote.h"
#include "StaticRange.h"
#include "Array.h"
#include "LN.h"

namespace Internal {
    template<class T>
    constexpr T next_pow2_impl( T a, T trial ) {
        return a <= trial ? trial : next_pow2_impl( a, 2 * trial );
    }

    template<class T>
    constexpr T ilog2_impl( T a, T expo ) {
        T trial = 1 << expo;
        return a <= trial ? expo : ilog2_impl( a, expo + 1 );
    }
}

template<class T,class U>
constexpr T ipow( T a, U b ) {
    return b ? a * ipow( a, b - 1 ) : 1;
}


template<class T,class U>
constexpr T ceil( T a, U m ) {
    return m ? ( a >= 0 ? a + m - 1 : a ) / m * m : a;
}

template<class T,class U>
constexpr T ceil_pow2( T a, U m ) {
    return ( ( a + ipow( 2, m ) - 1 ) >> m ) << m;
}

template<class T>
constexpr T next_pow2( T a ) {
    return a ? Internal::next_pow2_impl( a, T( 1 ) ) : 0;
}

template<class T>
constexpr T ilog2( T a ) {
    return Internal::ilog2_impl( a, T( 0 ) );
}

template<class T,class U>
constexpr T gcd( T a, U b ) {
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
constexpr T lcm( T a, U b ) {
    return a * b / gcd( a, b );
}


template<class T>
constexpr T pos_part( T val ) {
    return val >= 0 ? val : 0;
}

template<class TV>
auto norm_1( const TV &vec ) -> typename std::decay<decltype( vec[ 0 ] )>::type {
    using T = typename std::decay<decltype( vec[ 0 ] )>::type;
    T res = 0;
    for( const auto &v : vec )
        res += std::abs( v );
    return res;
}


template<class TV>
auto norm_2_pow_2( const TV &vec ) -> typename std::decay<decltype( vec[ 0 ] )>::type {
    using T = typename std::decay<decltype( vec[ 0 ] )>::type;
    T res = 0;
    for( const auto &v : vec )
        res += std::pow( v, 2 );
    return res;
}

template<class TV>
auto norm_2( const TV &vec ) -> typename std::decay<decltype( vec[ 0 ] )>::type {
    return std::sqrt( norm_2_pow_2( vec ) );
}

template<class T,size_t s>
std::array<T,s> operator+( const std::array<T,s> &a, const T &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] + b;
    return res;
}

template<class T,size_t s,int... b>
std::array<T,s> operator+( const std::array<T,s> &a, LN<b...> ln ) {
    std::array<T,s> res;
    StaticRange<s>::for_each( [&]( auto d ) {
        if ( d.val < s )
            res[ d.val ] = a[ d.val ] + ln.at( d );
    } );
    return res;
}

template<class T,size_t s>
std::array<T,s> operator+( const std::array<T,s> &a, const std::array<T,s> &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] + b[ i ];
    return res;
}

template<class T,size_t s>
std::array<T,s> operator-( const std::array<T,s> &a, const std::array<T,s> &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] - b[ i ];
    return res;
}


template<class T,size_t s>
std::array<T,s> operator-( const std::array<T,s> &a, const T &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] - b;
    return res;
}

template<class T,size_t s,int... b>
std::array<T,s> operator-( const std::array<T,s> &a, LN<b...> ln ) {
    std::array<T,s> res;
    StaticRange<s>::for_each( [&]( auto d ) {
        if ( d.val < s )
            res[ d.val ] = a[ d.val ] - ln.at( d );
    } );
    return res;
}

template<class T,size_t s,class U>
std::array<T,s> operator>>( const std::array<T,s> &a, const U &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] >> b;
    return res;
}

template<class T,size_t s,class U>
std::array<T,s> operator<<( const std::array<T,s> &a, const U &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] << b;
    return res;
}

template<class T,size_t s,class U>
std::array<typename TypePromote<T,U>::type,s> operator*( const std::array<T,s> &a, const U &b ) {
    std::array<typename TypePromote<T,U>::type,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] * b;
    return res;
}

template<class T,size_t s,class U>
std::array<typename TypePromote<T,U>::type,s> operator/( const std::array<T,s> &a, const U &b ) {
    std::array<typename TypePromote<T,U>::type,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = a[ i ] / b;
    return res;
}


template<class T>
T sum( const std::vector<T> &vec ) {
    T res = 0;
    for( const T &val : vec )
        res += val;
    return res;
}

template<class T>
T prod( const std::vector<T> &vec ) {
    T res = 1;
    for( const T &val : vec )
        res *= val;
    return res;
}

template<class T,std::size_t s>
T prod( const std::array<T,s> &vec ) {
    T res = 1;
    for( const T &val : vec )
        res *= val;
    return res;
}

template<class T>
T min( const std::vector<T> &vec ) {
    T res = std::numeric_limits<T>::max();
    for( const T &val : vec )
        res = std::min( res, val );
    return res;
}

template<class T>
T max( const std::vector<T> &vec ) {
    T res = std::numeric_limits<T>::min();
    for( const T &val : vec )
        res = std::max( res, val );
    return res;
}

template<class T,size_t s>
bool all_sup_eq( const std::array<T,s> &a, const std::array<T,s> &b ) {
    for( size_t i = 0; i < s; ++i )
        if ( a[ i ] < b[ i ] )
            return false;
    return true;
}

template<class T,size_t s>
bool all_sup( const std::array<T,s> &a, const std::array<T,s> &b ) {
    for( size_t i = 0; i < s; ++i )
        if ( a[ i ] <= b[ i ] )
            return false;
    return true;
}

template<class T,size_t s>
bool all_inf_eq( const std::array<T,s> &a, const std::array<T,s> &b ) {
    for( size_t i = 0; i < s; ++i )
        if ( a[ i ] > b[ i ] )
            return false;
    return true;
}

template<class T,size_t s>
bool all_inf( const std::array<T,s> &a, const std::array<T,s> &b ) {
    for( size_t i = 0; i < s; ++i )
        if ( a[ i ] >= b[ i ] )
            return false;
    return true;
}


template<class T,size_t s>
std::array<T,s> min( const std::array<T,s> &a, const std::array<T,s> &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = std::min( a[ i ], b[ i ] );
    return res;
}

template<class T,size_t s>
std::array<T,s> max( const std::array<T,s> &a, const std::array<T,s> &b ) {
    std::array<T,s> res;
    for( size_t i = 0; i < s; ++i )
        res[ i ] = std::max( a[ i ], b[ i ] );
    return res;
}

template<class T,class V>
void update_min_max( T &min, T &max, const V &val ) {
    if ( min > val ) min = val;
    if ( max < val ) max = val;
}


template<class T>
T atan2p( T y, T x ) {
    T res = std::atan2( y, x );
    return res < 0 ? res + 2 * M_PI : res;
}
