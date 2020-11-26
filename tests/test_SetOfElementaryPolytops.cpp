//#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
//#include "../src/sdot/geometry/Point.h"
//#include <parex/containers/Tensor.h>
//#include <parex/containers/Vec.h>
//#include <parex/Scheduler.h>
//#include <cmath>

//using namespace sdot;

//using Pt = Point<double,2>;
//using TI = std::uint64_t;
//using TF = Pt::TF;

//void test_triangle( TI dim = 2, TI nb_triangles = 15 ) {
//    SetOfElementaryPolytops sp( dim );
//    scheduler.log = true;

//    // construct
//    sp.add_repeated( "3", nb_triangles,
//        Vec<TF>{ 0, 0, 1, 0, 0, 1 },
//        Vec<TI>{ 0, 1, 2 }
//    );

//    // cut
//    Tensor<TF> normals( { nb_triangles, dim } );
//    Vec<TF> scalar_products( nb_triangles );
//    Vec<TI> new_face_ids( nb_triangles );
//    for( std::size_t i = 0; i < nb_triangles; ++i ) {
//        TF a = 2 * M_PI * i / nb_triangles;
//        Pt p = { std::cos( a ), std::sin( a ) };

//        scalar_products[ i ] = dot( p, Pt{ 0.33, 0.33 } );
//        normals.ptr( 0 )[ i ] = p[ 0 ];
//        normals.ptr( 1 )[ i ] = p[ 1 ];
//        new_face_ids[ i ] = 100 + i;
//    }

//    sp.plane_cut( normals, scalar_products, new_face_ids );
//    sp.display_vtk( "cut.vtk" );
//    P( sp );
//}

//void test_quad( TI dim = 2, TI nb_quads = 36 ) {
//    scheduler.kernel_code.compilation_flags = "";
//    scheduler.log = true;

//    // construct
//    SetOfElementaryPolytops sp( dim );
//    sp.add_repeated( "4", nb_quads,
//        Vec<TF>{ 0, -1, 1, 1, 0, -2, -1, 1 }, // positions
//        Vec<TI>{ 0, 1, 2, 3 } // face ids
//    );

//    // cut
//    Tensor<TF> normals( { nb_quads, dim } );
//    Vec<TF> scalar_products( nb_quads );
//    Vec<TI> new_face_ids( nb_quads );
//    for( std::size_t i = 0; i < nb_quads; ++i ) {
//        TF a = 2 * M_PI * i / nb_quads;
//        Pt p = { std::cos( a ), std::sin( a ) };

//        scalar_products[ i ] = dot( p, Pt{ 0.01, 0.0 } );
//        normals.ptr( 0 )[ i ] = p[ 0 ];
//        normals.ptr( 1 )[ i ] = p[ 1 ];
//        new_face_ids[ i ] = 100 + i;
//    }

//    sp.plane_cut( normals, scalar_products, new_face_ids );
//    sp.display_vtk( "cut.vtk" );
//    P( sp );
//}


//int main() {
//    //    test_triangle();
//    test_quad();
//}
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <parex/P.h>

int main() {
    //    test_triangle();
    //    test_quad();
    xt::xtensor<size_t, 1> a = xt::arange<size_t>(16);
    P( a );
}
