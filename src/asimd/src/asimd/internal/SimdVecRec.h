#pragma once

#include "../SimdVec.h"

namespace asimd {

/**
  contains a SimdVec<TF,s>, a SimdVec<TF,s/2>, ... until SimdVec<TF,1>
*/
template<class TF,int s>
struct SimdVecRec {
    /**/                    SimdVecRec     ( TF value ) : vec( value ), nxt( value ) {}
    /**/                    SimdVecRec     () {}

    template<int t> auto&   operator[]     ( N<t> n ) { return nxt[ n ]; }
    auto&                   operator[]     ( N<s> ) { return vec; }

    template<class F> void  for_each_scalar( const F &f ) { for( int i = 0; i < s; ++i ) f( vec[ i ] ); nxt.for_each_scalar( f ); }
    void                    write_to_stream( std::ostream &os ) const { os << vec << " " << nxt; }
    void                    assign_iota    ( TF beg, TF mul ) { vec = SimdVec<TF,s>::iota( beg, mul ); nxt.assign_iota( beg, mul ); }

    static SimdVecRec<TF,s> iota( TF beg, TF mul ) { SimdVecRec<TF,s> res; res.assign_iota( beg, mul ); return res; }

    SimdVec<TF,s>           vec;
    SimdVecRec<TF,s/2>      nxt;
};

template<class TF>
struct SimdVecRec<TF,0> {
    /**/                   SimdVecRec     ( TF ) {}
    /**/                   SimdVecRec     () {}

    template<class F> void for_each_scalar( const F & ) {}
    void                   write_to_stream( std::ostream & ) const {}
    void                   assign_iota    ( TF, TF ) {}
};

} // namespace asimd
