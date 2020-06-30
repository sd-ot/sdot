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
        Rp::Node{ {  0,  0,  0 }, 0 },
        Rp::Node{ { 12,  0,  0 }, 1 },
        Rp::Node{ {  0, 12,  0 }, 2 },
        Rp::Node{ {  0,  0, 12 }, 3 },
        Rp::Node{ { 12,  0, 12 }, 4 },
    } );

    P( rp );
    P( rp.measure() );

    //    std::vector<Rp> nrps;
    //    rp.plane_cut( nrps, Pt{ 4, 4, 4 }, Pt{ 1, 0, 0 } );

    //    VtkOutput vo;
    //    for( const Rp &nrp : nrps ) {
    //        nrp.display_vtk( vo );
    //        P( nrp );
    //    }
    //    vo.save( "out.vtk" );
}

int main() {
    test_2D();
    test_3D();
}
