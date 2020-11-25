#include <parex/arch/SimdVecRec.h>
#include <parex/containers/Vec.h>
#include <parex/arch/SimdRange.h>
#include <parex/support/P.h>
using namespace parex;

template<class TI,class A>
Vec<TI> *count_int_lane( const Vec<TI,A> &values, TI max_value ) {
    constexpr int max_nb_lanes = SimdSize<TI>::value;

    Vec<TI> *res = new Vec<TI>( max_value * max_nb_lanes, 0 );
    TI *count = res->ptr();

    SimdVecRec<TI,max_nb_lanes> off_lanes = SimdVecRec<TI,max_nb_lanes>::iota( 0, max_value );
    SimdRange<max_nb_lanes>::for_each( values.size(), [&]( TI index, auto s ) {
        using VI = SimdVec<TI,s.value>;

        VI value = VI::load( values.ptr() + index );
        VI oc = off_lanes[ s ] + value;

        VI::scatter( count, oc, VI::gather( count, oc ) + 1 );
    } );

    return res;
}
