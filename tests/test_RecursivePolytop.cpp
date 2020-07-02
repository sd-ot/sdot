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

    Rp rp{
        Pt{  0,  0,  0 },
        Pt{ 10,  0,  0 },
        Pt{  0, 10,  0 },
        Pt{ 10, 10,  0 },
        Pt{  0,  0, 10 },
        Pt{ 10,  0, 10 },
        Pt{  0, 10, 10 },
        Pt{ 10, 10, 10 }
    };

    rp.make_convex_hull();

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
    test_2D();
    test_3D();
}
