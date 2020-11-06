#pragma once

#include "../support/N.h"

namespace parex {

/**
*/
template<int s>
struct SimdRange {
    template<class TI,class F>
    static void for_each( TI end, const F &func ) {
        for_each_al<TI>( 0, end, func );
    }

    template<class TI,class F>
    static void for_each_al( TI beg, TI end, const F &func ) { ///< al means that beg is aligned
        for( TI cur = beg, nxt; ; cur = nxt ) {
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
    template<class TI,class F>
    static void for_each( TI end, const F &func ) {
        for_each_al<TI>( 0, end, func );
    }

    template<class TI,class F>
    static void for_each_al( TI beg, TI end, const F &func ) { ///< al means that beg is aligned
        for( TI cur = beg; cur < end; ++cur )
            func( cur, N<1>() );
    }
};

} // namespace parex
