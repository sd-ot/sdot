#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "catch_main.h"
using namespace parex;
using namespace sdot;

void test_triangle( int dim = 2, std::size_t nb_triangles = 5 ) {
    ElementaryPolytopTypeSet epts( dim );
    SetOfElementaryPolytops sp( epts, { /*.dst = MemoryGpu::gpu( 0 )*/ } );

    // construct
    sp.add_repeated( "3", nb_triangles, { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } }, { 0, 1, 2 } );

    Vector<> angles = Vector<>::linspace( 0, 2 * M_PI, nb_triangles, false );

    P( cos( angles ) );
    P( sin( angles ) );
    // Tensor<> normals = ;
    //    Vector<> scalar_products = ;
    //    Vector<> new_face_ids;
    //    sp.plane_cut( normals, scalar_products, new_face_ids );

    sp.display_vtk( "cut.vtk" );
    PN( sp );
    //    // cut
    //    xtensor<TF,2> angles = linspace<TF>( 0, 2 * M_PI, nb_triangles, false );
    //    xtensor<TF,2> normals = hstack( xtuple( cos( angles ), sin( angles ) ) );
    //    P( normals.shape() );
    //xtensor<TF,2> normals = hstack( xtuple( cos( angles ), sin( angles ) ) );

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

}

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
//int main() {
//    //    test_triangle();
//    test_quad();
//}

int main() {
    test_triangle();
    //    test_quad();
    //    Value v = (Task *)new CompiledTaskWithGeneratedSrc( "random", {}, [&]( Src &src, SrcWriter &/*sw*/ ) {
    //        src.include_directories << "ext/xtensor/install/include";
    //        src.include_directories << "ext/xsimd/install/include";
    //        src.includes << "<parex/containers/xtensor.h>";

    //        src << "TaskOut<xarray<double>> generated_func() {\n";
    //        src << "    return new xarray<double>( arange( 10 ) );\n";
    //        src << "\n}";
    //    } );
}
