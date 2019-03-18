#pragma once


#include "LN.h"

template<class LnList,int value_to_prepend>
struct StaticLnListPreprender {
    enum {
        size = LnList::size
    };

    template<class TF> inline
    static void for_each( const TF &f ) {
        LnList::for_each( [&]( auto ln ) {
            f( ln.template prepend<value_to_prepend>() );
        } );
    }

    template<class TF> inline
    static bool for_each_cont( const TF &f ) {
        return LnList::for_each_cont( [&]( auto ln ) {
            return f( ln.template prepend<value_to_prepend>() );
        } );
    }

    template<class TF> inline
    static void for_each_with_cpt( const TF &f ) {
        LnList::for_each_with_cpt( [&]( auto ln, auto cpt ) {
            f( ln.template prepend<value_to_prepend>(), cpt );
        } );
    }

};
