#include "../src/sdot/geometry/RecursivePolytop.h"
#include "../src/sdot/geometry/VtkOutput.h"
#include "../src/sdot/support/Rational.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using TF = Rational;

//// nsmake cpp_flag -march=native

void test_2D() {
    //    using Rp = RecursivePolytop<TF,2>;

    //    Rp rp = Rp::convex_hull( {
    //        Rp::Node{ { 0, 0 }, 0 },
    //        Rp::Node{ { 1, 0 }, 1 },
    //        Rp::Node{ { 0, 1 }, 2 },
    //    } );

    //    P( rp );
    //    P( rp.measure() );
}

void test_3D() {
    using Rp = RecursivePolytop<TF,3>;
    using Pt = Rp::Pt;

    Rp rp = Rp::convex_hull( {
        Rp::Node{ Pt{  0,  0,  0 }, 0 },
        Rp::Node{ Pt{ 10,  0,  0 }, 1 },
        Rp::Node{ Pt{  0, 10,  0 }, 2 },
        Rp::Node{ Pt{ 10, 10,  0 }, 3 },
        Rp::Node{ Pt{  0,  0, 10 }, 4 },
        Rp::Node{ Pt{ 10,  0, 10 }, 5 },
        Rp::Node{ Pt{  0, 10, 10 }, 6 },
        Rp::Node{ Pt{ 10, 10, 10 }, 7 },
    } );

    //    P( rp );
    //    P( rp.measure() );

    std::deque<Rp> cuts;
    rp.plane_cut( cuts, Pt{ 2, 2, 2 }, Pt{ 1, 0, 0 } );

    VtkOutput vo;
    for( const Rp &rp : cuts )
        rp.display_vtk( vo );
    vo.save( "out.vtk" );

    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

int main() {
    test_2D();
    test_3D();
}
