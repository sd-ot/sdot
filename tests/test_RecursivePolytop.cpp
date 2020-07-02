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
    using Rp = RecursivePolytop<TF,2>;
    using Pt = Rp::Pt;
    using TF = Rp::TF;

    //    std::vector<Pt> pts;
    //    for( TI i = 0, n = 5; i < n; ++i ) {
    //        double a = 2 * M_PI * i / n;
    //        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ) } );
    //    }

    //    Rp rp( pts );
    //    rp.make_convex_hull();
    //    rp.vertex( 0 ).pos[ 0 ] = -10;

    std::vector<Pt> pts;
    pts.push_back( {  0,  0 } );
    pts.push_back( { 10,  0 } );
    pts.push_back( {  0, 10 } );

    Rp rp( pts );
    rp.make_convex_hull();

    P( rp.measure() );
    P( rp.contains( Pt{ 2, 2 } ) );
    P( rp.contains( Pt{ 200, 0 } ) );

    //    Rp np = rp.plane_cut( Pt{ 0, 0 }, Pt{ 1, 0 } );
    //    P( np );

    //    VtkOutput vo;
    //    np.display_vtk( vo );
    //    vo.save( "out.vtk" );

    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

void test_3D() {
    //    using Rp = RecursivePolytop<TF,3>;
    //    using Pt = Rp::Pt;

    //    //    std::vector<Pt> pts;
    //    //    for( TI i = 0, n = 5; i < n; ++i ) {
    //    //        double a = 2 * M_PI * i / n;
    //    //        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ),   0 } );
    //    //        pts.push_back( { int( 100 * cos( a ) ), int( 100 * sin( a ) ), 100 } );
    //    //    }

    //    //    Rp rp( pts );
    //    //    rp.make_convex_hull();

    //    //    rp.vertex( 0 ).pos[ 0 ] = -10;
    //    //    rp.vertex( 1 ).pos[ 0 ] = -10;

    //    std::vector<Pt> pts;
    //    pts.push_back( { 0, 0, 0 } );
    //    pts.push_back( { 1, 0, 0 } );
    //    pts.push_back( { 0, 1, 0 } );
    //    pts.push_back( { 1, 1, 0 } );
    //    pts.push_back( { 0, 0, 1 } );
    //    pts.push_back( { 1, 0, 1 } );
    //    pts.push_back( { 0, 1, 1 } );
    //    pts.push_back( { 1, 1, 1 } );

    //    Rp rp( pts );
    //    rp.make_convex_hull();

    //    // P( rp );
    //    P( rp.measure() );

    //    Rp np = rp.plane_cut( Pt{ TF( 1 ) / 3, TF( 1 ) / 2, TF( 1 ) / 2 }, Pt{ 3, 2, 1 } );
    //    P( np.measure() );

    //    VtkOutput vo;
    //    np.display_vtk( vo );
    //    vo.save( "out.vtk" );

    //    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
    //    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

void test_4D() {
    //    using Rp = RecursivePolytop<TF,4>;
    //    using Pt = Rp::Pt;

    //    std::vector<Pt> pts;
    //    //    pts.push_back( { 0, 0, 0, 0 } );
    //    //    pts.push_back( { 2, 0, 0, 0 } );
    //    //    pts.push_back( { 0, 2, 0, 0 } );
    //    //    pts.push_back( { 2, 2, 0, 0 } );
    //    //    pts.push_back( { 0, 0, 2, 0 } );
    //    //    pts.push_back( { 2, 0, 2, 0 } );
    //    //    pts.push_back( { 0, 2, 2, 0 } );
    //    //    pts.push_back( { 2, 2, 2, 0 } );

    //    //    int a = 0, b = 2;
    //    //    pts.push_back( { a, a, a, 1 } );
    //    //    pts.push_back( { b, a, a, 1 } );
    //    //    pts.push_back( { a, b, a, 1 } );
    //    //    pts.push_back( { b, b, a, 1 } );
    //    //    pts.push_back( { a, a, b, 1 } );
    //    //    pts.push_back( { b, a, b, 1 } );
    //    //    pts.push_back( { a, b, b, 1 } );
    //    //    pts.push_back( { b, b, b, 1 } );
    //    pts.push_back( { 0, 0, 0, 0 } );
    //    pts.push_back( { 1, 0, 0, 0 } );
    //    pts.push_back( { 0, 1, 0, 0 } );
    //    pts.push_back( { 0, 0, 1, 0 } );
    //    pts.push_back( { 0, 0, 0, 1 } );

    //    Rp rp( pts );
    //    rp.make_convex_hull();

    //    // P( rp );
    //    P( rp.measure() );

    //    TF m = TF( 1 ) / 5;
    //    Rp n0 = rp.plane_cut( Pt{ m, m, m }, Pt{ +1, +2, +3 } );
    //    Rp n1 = rp.plane_cut( Pt{ m, m, m }, Pt{ -1, -2, -3 } );
    //    P( n0.measure() );
    //    P( n1.measure() );
    //    P( n0.measure() + n1.measure() );

    //    VtkOutput vo;
    //    n0.display_vtk( vo );
    //    n1.display_vtk( vo );
    //    vo.save( "out.vtk" );

    //    //    P( rp.contains( Pt{ 2, 2, 2 } ) );
    //    //    P( nrp.contains( Pt{ 2, 2, 2 } ) );
}

int main() {
    test_1D();
    test_2D();
    test_3D();
    test_4D();
}
