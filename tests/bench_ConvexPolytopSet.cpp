#include "../src/sdot/ConvexPolytop/ConvexPolytopSet.h"
#include "../src/sdot/support/range.h"
#include "../src/sdot/support/Time.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

template<class Cp,class TF>
void __attribute__ ((noinline)) do_the_cuts( Cp &cp, const TF *dxs, const TF *dys, const TF *sps ) {
    cp.plane_cut( { dxs, dys }, sps );
}

int main() {
    using Cp = ConvexPolytopSet<2,2,float,std::uint32_t>; // ,CpuArch::AVX512
    using TI = Cp::TI;
    using TF = Cp::TF;
    using Pt = Cp::Pt;

    TI nb_polytops = 1 << 14;
    std::uint64_t best_time = std::numeric_limits<std::uint64_t>::max();

    for( TI trial = 0; trial < 5000; ++trial ) {
        Cp cp;
        for( TI i = 0; i < nb_polytops; ++i )
            cp.add_shape( "3", { Pt{ 0, 0 }, Pt{ 1, 0 }, Pt{ 0, 1 } }, i );
        std::vector<TF> dxs( nb_polytops ), dys( nb_polytops ), sps( nb_polytops );
        for( TI i = 0; i < nb_polytops; ++i ) {
            TF a = 2 * M_PI * i / nb_polytops;
            Pt d( cos( a ), sin( a ) ), p( 5, 5 );
            sps[ i ] = dot( p, d );
            dxs[ i ] = d[ 0 ];
            dys[ i ] = d[ 1 ];
        }

        std::uint64_t t0 = 0, t1 = 0;
        RDTSC_START( t0 );
        do_the_cuts( cp, dxs.data(), dys.data(), sps.data() );
        RDTSC_FINAL( t1 );
        if ( best_time > t1 - t0 )
            best_time = t1 - t0;
    }

    P( double( best_time ) / nb_polytops );
}
