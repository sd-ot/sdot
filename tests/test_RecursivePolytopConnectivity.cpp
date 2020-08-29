#include "../src/sdot/geometry/RecursiveConvexPolytop.h"
#include "../src/sdot/geometry/VtkOutput.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using TF = double;

//// nsmake cpp_flag -g3

void test_1D() {
    //    using Rc = RecursivePolytopConnectivity<TF,1>;
    //    using Rp = RecursiveConvexPolytop<TF,1>;
    //    using Pt = Rp::Pt;

    //    std::vector<Pt> pts{ Pt{ 0 }, Pt{ 10 }, Pt{ 20 } };
    //    Rp rp( pts );

    //    Rc rc;
    //    rp.for_each_item_rec( [&]( const auto &v ) { rc = v; }, N<1>() );
    //    P( rc );

    //    TI nb_points = 10;
    //    std::vector<Rc> ncs;
    //    std::vector<bool> outside{ 0, 1, 1 };
    //    std::vector<TI> new_points_per_edge( pts.size() * ( pts.size() - 1 ) / 2, 0 );
    //    rc.conn_cut( ncs, nb_points, new_points_per_edge.data(), outside );

    //    P( ncs );
}

void test_2D() {
    using Rc = RecursivePolytopConnectivity<TF,2>;
    using Rp = RecursiveConvexPolytop<TF,2>;
    using Pt = Rp::Pt;

    // house shape
    std::vector<Pt> pts{ { 0, 0 }, { 10, 0 }, { 10, 10 }, { 5, 15 }, { 0, 10 } };
    Rp rp( pts );
    P( rp );

    Rc rc;
    rp.for_each_item_rec( [&]( const auto &v ) { P( v ); rc = v; }, N<2>() );
    P( rc );

    //    TI nb_points = 10;
    //    std::vector<Rc> ncs;
    //    std::vector<bool> outside{ 0, 1, 1 };
    //    std::vector<TI> new_points_per_edge( pts.size() * ( pts.size() - 1 ) / 2, 0 );
    //    rc.conn_cut( ncs, nb_points, new_points_per_edge.data(), outside );

    //    P( ncs );
}

void test_3D() {
    //    using Rp = RecursiveConvexPolytop<TF,3>;
    //    using Pt = Rp::Pt;

    //    std::vector<Pt> pts;
    //    pts.push_back( {  0,  0,  0 } );
    //    pts.push_back( { 10,  0,  0 } );
    //    pts.push_back( {  0, 10,  0 } );
    //    pts.push_back( { 10, 10,  0 } );
    //    pts.push_back( {  0,  0, 10 } );
    //    pts.push_back( { 10,  0, 10 } );
    //    pts.push_back( {  0, 10, 10 } );
    //    pts.push_back( { 10, 10, 10 } );
    //    pts.push_back( {  5,  5, 15 } );

    //    Rp rp( pts );
    //    // P( rp.measure() );

    //    Rp np = rp.plane_cut( Pt{ 5, 5, 12 }, Pt{ 0, 0, 1 } );
    //    // P( np.measure() );

    //    VtkOutput vo;
    //    np.display_vtk( vo );
    //    vo.save( "out.vtk" );

    //    P( np.contains( Pt{ 4, 6, 28 } ) );
    //    P( np.contains( Pt{ 4, 6, 8 } ) );
}

int main() {
    test_1D();
    test_2D();
    test_3D();
    //    test_4D();
    //    test_2D_intersection();
}
