#include "../src/sdot/ConvexPolytop/ConvexPolytopSet.h"
#include "../src/sdot/support/range.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

//// nsmake cpp_flag -march=native

int main() {
    using Cp = ConvexPolytopSet<3,2,float,unsigned>;
    using TI = Cp::TI;
    using TF = Cp::TF;
    using Pt = Cp::Pt;
    TI nb_volumes = 1; // 24;

    Cp cp;
    for( TI i = 0; i < nb_volumes; ++i ) {
        cp.add_shape( "3", { Pt{ 0, 0, int( i ) }, Pt{ 20, 0, int( i ) }, Pt{ 0, 20, int( i ) } }, 2 * i + 0 );
        //cp.add_shape( "4", { Pt{ 30, 0, int( i ) }, Pt{ 50, 0, int( i ) }, Pt{ 30, 20, int( i ) }, Pt{ 50, 20, int( i ) } }, 2 * i + 1 );
    }
    PN( cp );

    std::vector<TF> dxs( 2 * nb_volumes ), dys( 2 * nb_volumes ), dzs( 2 * nb_volumes ), sps( 2 * nb_volumes );
    for( TI i = 0; i < nb_volumes; ++i ) {
        TF a = 2 * M_PI * i / nb_volumes;
        Pt d( cos( a ), sin( a ) ), p[ 2 ] = { Pt( 5, 5 ), Pt( 35, 5 ) };
        for( TI o = 0; o < 2; ++o ) {
            sps[ 2 * i + o ] = dot( p[ o ], d );
            dxs[ 2 * i + o ] = d[ 0 ];
            dys[ 2 * i + o ] = d[ 1 ];
            dzs[ 2 * i + o ] = 0;
        }
    }

    cp.plane_cut( { dxs.data(), dys.data(), dzs.data() }, sps.data() );
    PN( cp );

    std::vector<TF> measures( 2 * nb_volumes );
    cp.get_measures( measures.data() );
    P( measures );
    for( TI i = 0; i < nb_volumes; ++i )
        P( measures[ i ] + measures[ nb_volumes + i ] );

    VtkOutput vo;
    cp.display_vtk( vo );
    vo.save( "out.vtk" );
}
