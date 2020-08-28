#include "../src/sdot/geometry/SimplexSumPolytop.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

void test_1D() {
    //    using Sp = SimplexSumPolytop<1>;
    //    using Pt = Sp::Pt;

    //    Sp s( std::array<Pt,2>{ Pt{ 0 }, Pt{ 1 } } );
    //    P( s.measure() );
    //    P( s );

    //    s.plane_cut( { 0.5 }, { 1.0 } );
    //    P( s.measure() );
    //    P( s );
}

void test_2D() {
    using Sp = SimplexSumPolytop<2>;
    using Pt = Sp::Pt;

    Sp s( { Pt{ 0, 0 }, Pt{ 2, 0 }, Pt{ 0, 2 } } );
    P( s.measure() );
    PN( s );

    s.plane_cut( { 1, 0 }, { 1, 0 } );
    P( s.measure() );
    PN( s );

    VtkOutput vo;
    s.display_vtk( vo );
    vo.save( "out.vtk" );
}

void test_3D() {
//    using Sp = SimplexSumPolytop<3>;
//    using Pt = Sp::Pt;

//    Sp s( { Pt{ 0, 0, 0 }, Pt{ 2, 0, 0 }, Pt{ 0, 2, 0 }, Pt{ 0, 0, 2 } } );
//    P( s.measure() );
//    P( s );

//    s.plane_cut( { 0.5, 0.5, 0.5 }, { -1, -1, -1 } );
//    P( s.measure() );
//    P( s );

//    VtkOutput vo;
//    s.display_vtk( vo );
//    vo.save( "out.vtk" );
}

int main() {
    test_1D();
    test_2D();
    test_3D();
}

