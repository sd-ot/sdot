#pragma once

#include "generic_ostream_output.h"
#include <iostream>
#include <iomanip>
#include <mutex>

struct WithSep {
    const char* sep;
};

template<class OS,class T0>               void __my_print( OS &os, const char* curr_sep, const char* next_sep, const T0      &t0                      ) { os << curr_sep << t0 << std::endl; }
template<class OS,class T0,class... Args> void __my_print( OS &os, const char* curr_sep, const char* next_sep, const T0      &t0, const Args &...args ) { os << curr_sep << t0; __my_print( os, next_sep, next_sep, args... ); }
template<class OS,         class... Args> void __my_print( OS &os, const char* curr_sep, const char* next_sep, const WithSep &ws, const Args &...args ) { __my_print( os, "", ws.sep, args... ); }

template<class OS,class... Args> void ___my_print( OS &os, const char* str, const Args &...args ) {
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
    #define PC do { static int cpt = 0; PE( cpt++ ); } while ( false )
    #define PS( val ) static int cpt = 0; if ( cpt++ == val )
#endif

