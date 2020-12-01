#pragma once

#include "SimdVec.h"
#include "N.h"

namespace asimd {

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

    template<class TI,class F>
    static void for_each_with_iota( TI beg, TI end, const F &func, TI ini = 0 ) {
        using SV = SimdVec<TI,s>;
        SV v = SV::iota( ini );
        for( TI cur = beg, nxt; ; cur = nxt ) {
            nxt = cur + s;
            if ( nxt > end )
                return SimdRange<s/2>::for_each_with_iota( cur, end, func, v[ 0 ] );
            func( cur, v, N<s>() );
            v = v + s;
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

    template<class TI,class F>
    static void for_each_with_iota( TI beg, TI end, const F &func, TI ini = 0 ) { ///< al means that beg is aligned
        for( TI cur = beg; cur < end; ++cur, ++ini )
            func( cur, ini, N<1>() );
    }
};

} // namespace asimd
