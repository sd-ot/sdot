#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/support/range.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

//// nsmake cpp_flag -march=native

template<int dim,class Fms>
void test_with_shape( VtkOutput &vo, N<dim>, const Fms &fms, TI nb_volumes = 20, VtkOutput::Pt off = 0.0 ) {
    using Cp = SetOfElementaryPolytops<dim,dim,float,unsigned>;
    using TI = typename Cp::TI;
    using TF = typename Cp::TF;
    using Pt = typename Cp::Pt;

    Cp cp;
    fms( cp, nb_volumes );
    //PN( cp );

    std::vector<TF> dxs( nb_volumes ), dys( nb_volumes ), sps( nb_volumes );
    for( TI i = 0; i < nb_volumes; ++i ) {
        TF a = 2 * M_PI * i / nb_volumes;
        Pt d( cos( a ), sin( a ) );
        sps[ i ] = dot( Pt( 5, 5 ), d );
        dxs[ i ] = d[ 0 ];
        dys[ i ] = d[ 1 ];
    }

    cp.plane_cut( { dxs.data(), dys.data() }, sps.data() );
    // PN( cp );

    //    std::vector<TF> measures( nb_volumes );
    //    cp.get_measures( measures.data() );
    //    P( measures );
    //    if ( nb_volumes % 2 == 0 )
    //        for( TI i = 0; i < nb_volumes / 2; ++i )
    //            P( measures[ i ] + measures[ i + nb_volumes / 2 ] );

    cp.display_vtk( vo, [&]( TI id ) { return off + VtkOutput::Pt{ 0.0, 0.0, 1.0 * id }; } );
}

int main() {
    auto mk_4 = []( auto &cp, TI nb_volumes ) { cp.add_shape( "4", { { 0, 0 }, { 20, 0 }, { 20, 20 }, { 0, 20 } }, 0, nb_volumes ); };
    auto mk_3 = []( auto &cp, TI nb_volumes ) { cp.add_shape( "3", { { 0, 0 }, { 20, 0 }, { 0, 20 } }, 0, nb_volumes ); };

    VtkOutput vo;
    test_with_shape( vo, N<2>(), mk_3, 20, {  0.0, 0.0, 0.0 } );
    // test_with_shape( vo, N<2>(), mk_4, 20, { 30.0, 0.0, 0.0 } );
    vo.save( "out.vtk" );
}
