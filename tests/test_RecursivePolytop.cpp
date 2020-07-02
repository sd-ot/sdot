#include "../src/sdot/geometry/RecursivePolytop.h"
#include "../src/sdot/geometry/VtkOutput.h"
#include "../src/sdot/support/Rational.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using TF = Rational;

//// nsmake cpp_flag -march=native

void test_1D() {
    //    using Rp = RecursivePolytop<TF,1>;
    //    using Pt = Rp::Pt;
    //    using TF = Rp::TF;

    //    Rp rp( { Pt{ 0 }, Pt{ 20 }, Pt{ 10 } } );
    //    rp.make_convex_hull();

    //    P( rp );
    //    P( rp.measure() );

    //    Rp np = rp.plane_cut( Pt{ 5 }, Pt{ 1 } );
    //    P( np );

    //    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
    //    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

void test_2D() {
//    using Rp = RecursivePolytop<TF,2>;
//    using Pt = Rp::Pt;
//    using TF = Rp::TF;

//    std::vector<Pt> pts;
//    for( TI i = 0, n = 5; i < n; ++i ) {
//        double a = 2 * M_PI * i / n;
//        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ) } );
//    }

//    Rp rp( pts );
//    rp.make_convex_hull();
//    rp.vertex( 0 ).pos[ 0 ] = -10;

//    //    P( rp );
//    P( rp.measure() );

//    Rp np = rp.plane_cut( Pt{ 0, 0 }, Pt{ 1, 0 } );
//    P( np );

//    VtkOutput vo;
//    np.display_vtk( vo );
//    vo.save( "out.vtk" );

//    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
//    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

void test_3D() {
    using Rp = RecursivePolytop<TF,3>;
    using Pt = Rp::Pt;

    std::vector<Pt> pts;
    for( TI i = 0, n = 5; i < n; ++i ) {
        double a = 2 * M_PI * i / n;
        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ),   0 } );
        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ), 100 } );
    }

    Rp rp( pts );
    rp.make_convex_hull();

    rp.vertex( 0 ).pos[ 0 ] = -10;
    rp.vertex( 1 ).pos[ 0 ] = -10;

    //    P( rp );
    P( rp.measure() );

    Rp np = rp.plane_cut( Pt{ 2, 2, 2 }, Pt{ 1, 0, 0 } );
    P( np );

    VtkOutput vo;
    np.display_vtk( vo );
    vo.save( "out.vtk" );

    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

int main() {
    test_1D();
    test_2D();
    test_3D();
}
