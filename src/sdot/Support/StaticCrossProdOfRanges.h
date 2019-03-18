#pragma once

#include "StaticRange.h"
#include "Math.h"
#include "LN.h"
#include <tuple>

/**
  gives things like
    LN<0,0>
    LN<1,0>
    LN<2,0>
    LN<0,1>
    LN<1,1>
    LN<2,1>
    ...

    _a, _b and _c work like in StaticRange
*/
template<int dim,int _a,int _b=_a-1,int _c=1>
class StaticCrossProdOfRanges {
public:
    enum {
        size = ipow( StaticRange<_a,_b,_c>::size + 0, dim )
    };

    template<class TF> inline
    static void for_each( const TF &f ) {
        _for_each( f, LN<>() );
    }

    template<class TF> inline
    static bool for_each_cont( const TF &f ) {
        bool cont = true;
        _for_each_cont( f, LN<>(), cont );
        return cont;
    }

    template<class TF> inline
    static void for_each_with_cpt( const TF &f ) {
        _for_each_with_cpt( f, LN<>(), N<0>() );
    }

    /// { f( LN<a,b,...> ) for each a,b,...  }
    template<class TF> inline
    static auto make_tuple( const TF &f ) {
        return _make_tuple( f, N<0>() );
    }


    /// { f( LN<a,b,...>, N<cpt> ) for each a,b,...  }
    template<class TF> inline
    static auto make_tuple_with_cpt( const TF &f ) {
        return _make_tuple_with_cpt( f, N<0>() );
    }

    ///  op( op( f( LN<a0,b0,...> ), LN<a1,b10,...> ), ... )
    template<class TO,class TF> inline
    static auto reduction( const TO &op, const TF &f ) {
        return _reduction( op, f, N<size-1>() );
    }

    template<class TF,class LN> inline
    static void _for_each( const TF &f, LN ) {
        StaticRange<_a,_b,_c>::for_each( [&]( auto n ) {
            StaticCrossProdOfRanges<dim-1,_a,_b,_c>::_for_each( f, typename LN::template Prepend<n.val>::T() );
        } );
    }

    template<class TF,class LN> inline
    static void _for_each_cont( const TF &f, LN, bool &cont ) {
        StaticRange<_a,_b,_c>::for_each( [&]( auto n ) {
            StaticCrossProdOfRanges<dim-1,_a,_b,_c>::_for_each_cont( f, typename LN::template Prepend<n.val>::T(), cont );
        } );
    }

    template<class TF,class LN,int cpt> inline
    static void _for_each_with_cpt( const TF &f, LN, N<cpt> ) {
        using SR = StaticRange<_a,_b,_c>;
        SR::for_each_with_cpt( [&]( auto n, auto cn ) {
            StaticCrossProdOfRanges<dim-1,_a,_b,_c>::_for_each_with_cpt( f, typename LN::template Prepend<n.val>::T(), N<cpt+ipow(int(SR::len),dim-1)*cn.val>() );
        } );
    }

    template<int num>
    static auto ln_for_cpt( N<num> ) {
        using SR = StaticRange<_a,_b,_c>;
        return StaticCrossProdOfRanges<dim-1,_a,_b,_c>::ln_for_cpt( N<num/SR::size>() ).prepend( N<SR::beg+num%SR::size*SR::inc>() );
    }

private:
    template<class TF> inline
    static auto _make_tuple( const TF &f, N<size> ) {
        return std::tuple<>();
    }
    template<class TF,int cpt> inline
    static auto _make_tuple( const TF &f, N<cpt> n_cpt ) {
        return std::tuple_cat( std::make_tuple( f( ln_for_cpt( n_cpt ) ) ), _make_tuple( f, N<cpt+1>() ) );
    }

    template<class TF> inline
    static auto _make_tuple_with_cpt( const TF &f, N<size> ) {
        return std::tuple<>();
    }
    template<class TF,int cpt> inline
    static auto _make_tuple_with_cpt( const TF &f, N<cpt> n_cpt ) {
        return std::tuple_cat( std::make_tuple( f( ln_for_cpt( n_cpt ), n_cpt ) ), _make_tuple_with_cpt( f, N<cpt+1>() ) );
    }

    template<class OP,class TF> inline
    static auto _reduction( const OP &op, const TF &f, N<0> n_cpt ) {
        return f( ln_for_cpt( n_cpt ) );
    }
    template<class OP,class TF,int cpt> inline
    static auto _reduction( const OP &op, const TF &f, N<cpt> n_cpt ) {
        return op( _reduction( op, f, N<cpt-1>() ), f( ln_for_cpt( n_cpt ) ) );
    }
};

template<int _a,int _b,int _c>
class StaticCrossProdOfRanges<0,_a,_b,_c> {
public:
    enum {
        size = 1
    };

    template<class TF> inline
    static void for_each( const TF &f ) {
        f( LN<>() );
    }

    template<class TF> inline
    static bool for_each_cont( const TF &f ) {
        return f( LN<>() );
    }

    template<class TF> inline
    static void for_each_with_cpt( const TF &f ) {
        f( LN<>(), N<0>() );
    }

    template<class TF> inline
    static auto make_tuple( const TF &f ) {
        return std::make_tuple( f( LN<>() ) );
    }

    template<class TF> inline
    static auto make_tuple_with_cpt( const TF &f ) {
        return std::make_tuple( f( LN<>(), N<0>() ) );
    }

    template<int num>
    static LN<> ln_for_cpt( N<num> ) {
        return {};
    }

    ///  f( LN<> )
    template<class TO,class TF> inline
    static auto reduction( const TO &op, const TF &f ) {
        return f( LN<>() );
    }

    template<class TF,class LN> inline
    static void _for_each( const TF &f, LN ln ) {
        f( ln );
    }

    template<class TF,class LN> inline
    static void _for_each_cont( const TF &f, LN ln, bool &cont ) {
        if ( cont )
            cont = f( ln );
    }

    template<class TF,class LN,int cpt> inline
    static void _for_each_with_cpt( const TF &f, LN ln, N<cpt> nc ) {
        f( ln, nc );
    }
};

