#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/geometry/Point.h"
#include "../src/parex/support/P.h"
#include <cmath>

using namespace parex;
using namespace sdot;

using Pt = Point<double,2>;
using TI = std::uint64_t;
using TF = Pt::TF;

void test_triangle() {
    const int dim = 2;
    SetOfElementaryPolytops sp( dim );

    // construct
    TI nb_triangles = 15;
    sp.add_repeated( triangle(), nb_triangles,
        Tensor<double>( Vec<TI>{ nb_triangles, dim }, { 0, 1, 0,  0, 0, 1 } ),
        Vec<double>( { 0, 1, 2 } ),
        100
    );

    sp.add_repeated( triangle(), nb_triangles,
        Tensor<double>( Vec<TI>{ nb_triangles, dim }, { 0, 1, 0,  0, 0, 1 } ),
        Vec<double>( { 0, 1, 2 } ),
        200
    );

    P( sp );

    //    // cut
    //    std::vector<TF> cxs, cys, css;
    //    std::vector<TI> new_face_ids;
    //    for( std::size_t i = 0; i < nb_triangles; ++i ) {
    //        TF a = 2 * M_PI * i / nb_triangles;
    //        Pt p = { std::cos( a ), std::sin( a ) };
    //        cxs.push_back( p[ 0 ] );
    //        cys.push_back( p[ 1 ] );
    //        css.push_back( dot( p, Pt{ 0.33, 0.33 } ) );
    //        new_face_ids.push_back( 100 + i );
    //    }

    //    sp.plane_cut( { { ks, cxs }, { ks, cys } }, { ks, css }, { ks, new_face_ids } );
    //    P( sp );

    //    // display
    //    std::vector<VtkOutput::Pt> off_vtk;
    //    for( std::size_t i = 0; i < nb_triangles; ++i )
    //        off_vtk.push_back( VtkOutput::Pt{ 0.0, 0.0, 0.2 * i } );

    //    VtkOutput vo;
    //    sp.display_vtk( vo, off_vtk.data() );
    //    vo.save( "cut.vtk" );
}

//void test_quad() {
//    std::vector<std::unique_ptr<KernelSlot>> ak = KernelSlot::available_slots( TypeName<double>::name(), TypeName<std::size_t>::name() );
//    KernelSlot *ks = ak[ 0 ].get();

//    SetOfElementaryPolytops sp( ks, 2 );

//    // construct
//    TI nb_quads = 36;
//    sp.add_repeated( quad(), nb_quads,
//        { ks, std::vector<TF>{ 0, -1, 1, 1, 0, -2, -1, 1 } }, // positions
//        { ks, std::vector<TI>{ 0, 1, 2, 3 } }, // face ids
//        0 // beg item id
//    );

//    // cut
//    std::vector<TF> cxs, cys, css;
//    std::vector<TI> new_face_ids;
//    for( std::size_t i = 0; i < nb_quads; ++i ) {
//        TF a = 2 * M_PI * i / nb_quads;
//        Pt p = { std::cos( a ), std::sin( a ) };
//        cxs.push_back( p[ 0 ] );
//        cys.push_back( p[ 1 ] );
//        css.push_back( dot( p, Pt{ 0.01, 0.0 } ) );
//        new_face_ids.push_back( 100 + i );
//    }

//    sp.plane_cut( { { ks, cxs }, { ks, cys } }, { ks, css }, { ks, new_face_ids } );
//    P( sp );

//    // display
//    std::vector<VtkOutput::Pt> off_vtk;
//    for( std::size_t i = 0; i < nb_quads; ++i )
//        off_vtk.push_back( VtkOutput::Pt{ 3.5 * ( i % 6 ), - 3.5 * ( i / 6 ), 0.0 } );

//    VtkOutput vo;
//    sp.display_vtk( vo, off_vtk.data() );
//    vo.save( "cut.vtk" );
//}

int main() {
    test_triangle();
    // test_quad();
}
