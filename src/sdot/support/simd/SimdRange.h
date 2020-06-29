#pragma once

#include "../N.h"

/**
*/
template<int s>
struct SimdRange {
    template<class F>
    static void for_each( int end, const F &func ) {
        for_each_al( 0, end, func );
    }

    template<class F>
    static void for_each_al( int beg, int end, const F &func ) {
        for( int cur = beg, nxt; ; cur = nxt ) {
            nxt = cur + s;
            if ( nxt > end )
                return SimdRange<s/2>::for_each_al( cur, end, func );
            func( cur, N<s>() );
        }
    }
};

//
template<>
struct SimdRange<1> {
    template<class F>
    static void for_each( int end, const F &func ) {
        for_each_al( 0, end, func );
    }

    template<class F>
    static void for_each_al( int beg, int end, const F &func ) {
        for( int cur = beg; cur < end; ++cur )
            func( cur, N<1>() );
    }
};
