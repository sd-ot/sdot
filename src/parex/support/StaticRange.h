#pragma once

#include "StaticGenericListOfNumbers.h"
#include <tuple>

namespace parex {

/**
*/
template<int _a,int _b=_a-1,int _c=1>
struct StaticRange : StaticGenericListOfNumbers<StaticRange<_a,_b,_c>> {
    enum {
        beg = ( _a <= _b ) ^ ( _c < 0 ) ? _a :  0,
        end = ( _a <= _b ) ^ ( _c < 0 ) ? _b : _a,
        inc = _c,
        len = ( end - beg ) / inc,
        size = len
    };

    template<class TF> inline
    static void for_each( const TF &f ) {
        for_each_in_range( f, parex::N<beg>(), parex::N<end>(), parex::N<inc>() );
    }

    template<class TF> inline
    static bool for_each_cont( const TF &f ) {
        bool cont = true;
        _for_each_cont( f, parex::N<beg>(), cont );
        return cont;
    }

    template<class TF> inline
    static void for_each_with_cpt( const TF &f ) {
        for_each_with_cpt_in_range( f, parex::N<beg>(), parex::N<end>(), parex::N<inc>(), parex::N<0>() );
    }

    /// { f( N<a>() for each a in range }
    template<class TF> inline
    static auto make_tuple( const TF &f ) {
        return _make_tuple( f, parex::N<0>() );
    }

    ///  op( op( f( LN<a0,b0,...> ), LN<a1,b10,...> ), ... )
    template<class TO,class TF> inline
    static auto reduction( const TO &op, const TF &f ) {
        return _reduction( op, f, parex::N<size-1>() );
    }

    template<int num>
    static auto n_for_cpt( parex::N<num> ) {
        return parex::N< beg + num * inc >();
    }

private:
    template<class TF> inline
    static auto _make_tuple( const TF &/*f*/, parex::N<size> ) {
        return std::tuple<>();
    }
    template<class TF,int cpt> inline
    static auto _make_tuple( const TF &f, parex::N<cpt> n_cpt ) {
        return std::tuple_cat( std::make_tuple( f( n_cpt ) ), _make_tuple( f, parex::N<cpt+1>() ) );
    }

    template<class TF> inline
    static void _for_each_cont( const TF &/*f*/, parex::N<end>, bool &cont ) {
    }
    template<class TF,int cur> inline
    static void _for_each_cont( const TF &f, parex::N<cur> n_cur, bool &cont ) {
        if ( cont ) {
            cont = f( n_cur );
            _for_each_cont( f, parex::N<cur+inc>(), cont );
        }
    }

    template<class OP,class TF> inline
    static auto _reduction( const OP &/*op*/, const TF &f, parex::N<0> n_cpt ) {
        return f( n_for_cpt( n_cpt ) );
    }
    template<class OP,class TF,int cpt> inline
    static auto _reduction( const OP &op, const TF &f, parex::N<cpt> n_cpt ) {
        return op( _reduction( op, f, parex::N<cpt-1>() ), f( n_for_cpt( n_cpt ) ) );
    }
};


//
template<class TF,int beg,int end,int inc> inline
void for_each_in_range( const TF &f, parex::N<beg> nbeg, parex::N<end> nend, parex::N<inc> ninc ) {
    f( nbeg );
    for_each_in_range( f, parex::N<beg+inc>(), nend, ninc );
}

template<class TF,int end,int inc> inline
void for_each_in_range( const TF &f, parex::N<end> /*nbeg*/, parex::N<end> /*nend*/, parex::N<inc> /*ninc*/ ) {
}

//
template<class TF,int beg,int end,int inc,int cpt> inline
void for_each_with_cpt_in_range( const TF &f, parex::N<beg> nbeg, parex::N<end> nend, parex::N<inc> ninc, parex::N<cpt> ncpt ) {
    f( nbeg, ncpt );
    for_each_with_cpt_in_range( f, parex::N<beg+inc>(), nend, ninc, parex::N<cpt+1>() );
}

template<class TF,int end,int inc,int cpt> inline
void for_each_with_cpt_in_range( const TF &f, parex::N<end> /*nbeg*/, parex::N<end> /*nend*/, parex::N<inc> /*ninc*/, parex::N<cpt> /*ncpt*/ ) {
}

}
