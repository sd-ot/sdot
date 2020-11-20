#include <parex/containers/Tensor.h>
#include <parex/arch/SimdVecRec.h>
#include <parex/arch/SimdRange.h>
#include <parex/support/N.h>
#include <array>

using namespace parex;

template<class TF,class TI>
std::array<TF,2> min_max_scalar( const TF *ptr, TI len ) {
    constexpr int ms = SimdSize<TF>::value;

    SimdVecRec<TF,ms> mi( + std::numeric_limits<TF>::max() );
    SimdVecRec<TF,ms> ma( - std::numeric_limits<TF>::max() );
    SimdRange<ms>::for_each( len, [&]( TI index, auto s ) {
        using VF = SimdVec<TF,s.value>;
        VF inp = VF::load( ptr + index );
        mi[ s ] = min( mi[ s ], inp );
        ma[ s ] = max( ma[ s ], inp );
    } );

    TF mif = + std::numeric_limits<TF>::max();
    TF maf = - std::numeric_limits<TF>::max();
    mi.for_each_scalar( [&]( TF v ) { using std::min; mif = min( mif, v ); } );
    ma.for_each_scalar( [&]( TF v ) { using std::max; maf = max( maf, v ); } );

    return { mif, maf };
}

template<class TF,class A,class TI,int dim>
Vec<std::array<TF,2>> *min_max_pd( const Tensor<TF,A,TI> &diracs, N<dim> ) {
    Vec<std::array<TF,2>> *res = new Vec<std::array<TF,2>>( dim );
    for( int d = 0; d < dim; ++d )
        res->operator[]( d ) = min_max_scalar( diracs.ptr( d ), diracs.x_size() );
    return res;
}
