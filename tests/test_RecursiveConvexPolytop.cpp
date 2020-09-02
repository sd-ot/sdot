#include "../src/sdot/geometry/RecursiveConvexPolytop.h"
#include "../src/sdot/geometry/VtkOutput.h"
#include "../src/sdot/support/Rational.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using TF = Rational;

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -g3

void test_1D_convex() {
//    using Rp = RecursiveConvexPolytop<TF,1>;
//    using Pt = Rp::Pt;

//    Rp rp( { Pt{ 0 }, Pt{ 10 }, Pt{ 20 } } );
//    P( rp );
}

void test_2D_convex() {
//    using Rp = RecursiveConvexPolytop<TF,2>;
//    using Pt = Rp::Pt;

//    // house shape
//    Rp rp( { { 0, 0 }, { 10, 0 }, { 10, 10 }/*, { 5, 15 }, { 0, 10 }*/ } );
//    //    P( rp.contains( Pt{ 2, 22 } ) );
//    //    P( rp.contains( Pt{ 2, 2 } ) );
//    //    P( rp.measure() );
//    P( rp );


//    //    Rp op = rp.plane_cut( Pt{ 5, 12 }, Pt{ 0, +1 } );
//    //    P( op );

//    VtkOutput vo;
//    rp.display_vtk( vo );
//    vo.save( "out.vtk" );
}

void test_3D_convex() {
    using Rp = RecursiveConvexPolytop<TF,3>;
    using Pt = Rp::Pt;

    std::vector<Pt> pts;
    pts.push_back( {  0,  0,  0 } );
    pts.push_back( { 10,  0,  0 } );
    pts.push_back( {  0, 10,  0 } );
    pts.push_back( { 10, 10,  0 } );
    pts.push_back( {  0,  0, 10 } );
    pts.push_back( { 10,  0, 10 } );
    pts.push_back( {  0, 10, 10 } );
    pts.push_back( { 10, 10, 10 } );
    pts.push_back( {  5,  5, 15 } );

    Rp rp( std::move( pts ) );
    // P( rp.measure() );
    P( rp );

    Rp np = rp.plane_cut( Pt{ 5, 5, 12 }, Pt{ 0, 0, 1 } );
    // P( np.measure() );

    VtkOutput vo;
    np.display_vtk( vo );
    vo.save( "out.vtk" );

    //    P( np.contains( Pt{ 4, 6, 28 } ) );
    //    P( np.contains( Pt{ 4, 6, 8 } ) );
}

void test_4D_convex() {
    //    using Rp = RecursiveConvexPolytop<TF,4>;
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

void test_1D_conn() {
    //    using Rp = RecursiveConvexPolytop<TF,1>;
    //    using Pt = Rp::Pt;

    //    Rp rp( { Pt{ 0 }, Pt{ 10 }, Pt{ 20 } } );
    //    P( rp.contains( Pt{ 22 } ) );
    //    P( rp.contains( Pt{ 2 } ) );
    //    P( rp );

    //    //    Rp np = rp.plane_cut( Pt{ 5 }, Pt{ 1 } );
    //    std::vector<std::vector<Rp>> np = rp.conn_cut( Pt{ 5 }, Pt{ 1 } );
    //    P( np );
}

void test_2D_conn() {
    //    using Rp = RecursiveConvexPolytop<TF,2>;
    //    using Pt = Rp::Pt;

    //    // house shape
    //    Rp rp( { { 0, 0 }, { 10, 0 }, { 10, 10 }, { 5, 15 }, { 0, 10 } } );
    //    P( rp.contains( Pt{ 2, 22 } ) );
    //    P( rp.contains( Pt{ 2, 2 } ) );
    //    P( rp );

    //    std::vector<std::vector<Rp>> np = rp.conn_cut( Pt{ 5, 12 }, Pt{ 0, +1 } );
    //    P( np );

    //    //    VtkOutput vo;
    //    //    op.display_vtk( vo );
    //    //    vo.save( "out.vtk" );
}

void test_3D_conn() {
}

void test_4D_conn() {
}

int main() {
    test_1D_convex();
    test_2D_convex();
    test_3D_convex();
    test_4D_convex();

    test_1D_conn();
    test_2D_conn();
    test_3D_conn();
    test_4D_conn();
}
