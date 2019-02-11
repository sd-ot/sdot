#pragma once

#include "TensorOrder.h"
#include "TypeConfig.h"
#include "EnableIf.h"
#include "N.h"
#include "S.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <mutex>

//
template<class C>
struct HasWriteToStream {
    template<class   T> static constexpr typename EnableIf<1,N<1>,decltype(&T::write_to_stream   )>::T has( S<T> ) { return {}; }
    template<class   T> static constexpr typename EnableIf<1,N<1>,decltype(T::has_write_to_stream)>::T has( S<T> ) { return {}; }
    template<class...T> static constexpr N<0>                                                          has( T... ) { return {}; }
    using V = decltype( has( S<C>() ) );
    enum { value = V::val };
};

// operator<<( ostream ) if we have a write to stream
template<class T>
typename EnableIf<HasWriteToStream<T>::value,std::ostream>::T &operator<<( std::ostream &os, const T &val ) {
    const_cast<T &>( val ).write_to_stream( os );
    return os;
}

// operator<<( ostream ) for tensor order == 1 if no write to stream
template<class T>
typename EnableIf<TensorOrder<T>::res == 1 && HasWriteToStream<T>::value == false, std::ostream>::T &operator<<( std::ostream &os, const T &val ) {
    if ( size_t s = val.size() ) {
        os << val[ 0 ];
        for( size_t i = 1; i < s; ++i )
            os << ' ' << val[ i ];
    }
    return os;
}

struct WithSep {
    const char *sep;
};

template<class OS,class T0>               void __my_print( OS &os, const char *curr_sep, const char *next_sep, const T0      &t0                      ) { os << curr_sep << t0 << std::endl; }
template<class OS,class T0,class... Args> void __my_print( OS &os, const char *curr_sep, const char *next_sep, const T0      &t0, const Args &...args ) { os << curr_sep << t0; __my_print( os, next_sep, next_sep, args... ); }
template<class OS,         class... Args> void __my_print( OS &os, const char *curr_sep, const char *next_sep, const WithSep &ws, const Args &...args ) { __my_print( os, "", ws.sep, args... ); }

template<class OS,class... Args> void ___my_print( OS &os, const char *str, const Args &...args ) {
    static std::mutex m;
    m.lock();
    __my_print( os, str, ", ", args... );
    os.flush();
    m.unlock();
}

#ifndef P
    #define P( ... ) \
        ___my_print( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
    #define PI( N, ... ) \
        ___my_print( std::cout, N " -> "            , __VA_ARGS__ )
    #define PE( ... ) \
        ___my_print( std::cerr, #__VA_ARGS__ " -> " , __VA_ARGS__ )
    #define PN( ... ) \
        ___my_print( std::cout, #__VA_ARGS__ " ->\n", __VA_ARGS__ )
    // PRINT with file and line info
    #define PM( ... ) \
        ___my_print( std::cout, #__VA_ARGS__ " -> ", __VA_ARGS__, WithSep{""}, " (", __FILE__, ':', __LINE__, ')' )
    // PRINT with counter
    // #define PC do { static int cpt = 0; PE( cpt++ ); } while ( false )
    #define PS( val ) static int cpt = 0; if ( cpt++ == val )
#endif

template<class T>
struct BinaryRepr {
    BinaryRepr( const T &val ) : val( val ) {
    }
    void write_to_stream( std::ostream &os ) const {
        for( size_t i = 8 * sizeof( val ); i--; )
            os << ( val & ( T( 1 ) << i ) ? '1' : '0' );
    }
    T val;
};

template<class T>
BinaryRepr<T> binary_repr( const T &val ) {
    return val;
}

template<class T,class ...Args>
std::string to_string( const T &val, const Args &...args ) {
    std::ostringstream ss;
    val.write_to_stream( ss, args... );
    return ss.str();
}

template<class T>
std::string to_string( const T &val ) {
    std::ostringstream ss;
    ss << val;
    return ss.str();
}

std::string to_string( const PI8 *val );

