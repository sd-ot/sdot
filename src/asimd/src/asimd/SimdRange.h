#pragma once

#include "internal/N.h"
#include "SimdVec.h"

namespace asimd {

/**
  next_size = size/2 => use of smaller SIMD instruction at the end before size 1 (might be faster in some cases, but needing more instructions)
*/
template<int size,int next_size=1>
struct SimdRange {
    /// func( TI index, N<simd_size> )
    /// Version with beg % size assumed to be 0
    template<class TI,class Func>
    static void for_each_with_beg_aligned( TI beg, TI end, Func &&func ) {
        for( TI cur = beg, nxt; ; cur = nxt ) {
            nxt = cur + size;
            if ( nxt > end )
                return SimdRange<next_size>::for_each_with_beg_aligned( cur, end, std::forward<Func>( func ) );

            func( cur, N<size>() );
        }
    }

    /// func( TI index, N<simd_size> )
    /// beg % size may be != 0. It it's the case,
    template<class TI,class Func>
    static void for_each( TI beg, TI end, Func &&func ) {
        // go to a aligned beg
        if ( TI mod = beg % size )
            for( TI mnd = std::min( beg + size - mod, end ); beg < mnd; ++beg )
                func( beg, N<1>() );
        for_each_with_beg_aligned( beg, end, std::forward<Func>( func ) );
    }
};

//
template<int next_size>
struct SimdRange<1,next_size> {
    template<class TI,class Func>
    static void for_each_with_beg_aligned( TI beg, TI end, Func &&func ) {
        for_each( beg, end, std::forward<Func>( func ) );
    }

    template<class TI,class Func>
    static void for_each( TI beg, TI end, Func &&func ) {
        for( TI cur = beg; cur < end; ++cur )
            func( cur, N<1>() );
    }
};

} // namespace asimd
