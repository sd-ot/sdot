#include "../../../support/P.h"
#include "ConvexHull.h"

//void test_1D() {
//    using Pt = ConvexHullIndices<1>::Pt;
//    std::vector<Pt> pts;
//    pts.push_back( { 0 } );
//    pts.push_back( { 2 } );
//    pts.push_back( { 1 } );

//    ConvexHullIndices<1> ch( pts );
//    PN( ch );
//    P( ch.measure( pts ) );
//}

//void test_2D() {
//    using Pt = ConvexHullIndices<2>::Pt;
//    std::vector<Pt> pts;
//    pts.push_back( { 0, 0 } );
//    pts.push_back( { 1, 0 } );
//    pts.push_back( { 0, 1 } );

//    ConvexHullIndices<2> ch( pts );
//    PN( ch );
//    P( ch.measure( pts ) );
//}

//void test_3D_Wedge() {
//    using Pt = ConvexHullIndices<3>::Pt;
//    using TI = ConvexHullIndices<3>::TI;
//    std::vector<Pt> pts;
//    pts.push_back( { 0, 0, 0 } );
//    pts.push_back( { 1, 0, 0 } );
//    pts.push_back( { 0, 1, 0 } );
//    pts.push_back( { 0, 0, 1 } );
//    pts.push_back( { 1, 0, 1 } );
//    pts.push_back( { 0, 1, 1 } );

//    ConvexHullIndices<3> ch( pts );
//    PNR( "/1", "", ch );
//    P( ch.measure( pts ) );

//    // make a the same thing with a permutation
//    std::vector<Pt> nts{ pts[ 1 ], pts[ 3 ], pts[ 0 ], pts[ 2 ], pts[ 5 ], pts[ 4 ] };
//    ConvexHullIndices<3> dh( nts );
//    P( dh );

//    std::vector<TI> perm_dh_to_ch( pts.size() );
//    P( dh.is_a_permutation_of( ch, perm_dh_to_ch.data() ) );
//    P( perm_dh_to_ch );
//}

//void test_3D_Pyramid() {
//    using Pt = ConvexHullIndices<3>::Pt;
//    using TI = ConvexHullIndices<3>::TI;
//    std::vector<Pt> pts;
//    pts.push_back( { 0, 0, 0 } );
//    pts.push_back( { 2, 0, 0 } );
//    pts.push_back( { 0, 2, 0 } );
//    pts.push_back( { 2, 2, 0 } );
//    pts.push_back( { 1, 1, 1 } );

//    ConvexHullIndices<3> ch( pts );
//    PNR( "/1", "", ch );
//    P( ch.measure( pts ) );

//    // make a the same thing with a permutation
//    std::vector<Pt> nts{ pts[ 1 ], pts[ 3 ], pts[ 0 ], pts[ 4 ], pts[ 2 ] };
//    ConvexHullIndices<3> dh( nts );

//    std::vector<TI> perm_dh_to_ch( pts.size() );
//    P( dh.is_a_permutation_of( ch, perm_dh_to_ch.data() ) );
//    P( perm_dh_to_ch );
//}

//void test_cut() {
//    using Pt = ConvexHull<2>::Pt;
//    ConvexHull<2> ch( {
//        Pt{  0,  0 },
//        Pt{ 10,  0 },
//        Pt{  0, 10 },
//    } );
//    ConvexHull<2> dh = ch.cut( Pt{ 8, 10 }, Pt{ 0, 1 } );
//    P( dh );
//}

//void test_intersection_2D() {
//    using Pt = ConvexHull<2>::Pt;
//    ConvexHull<2> ch( {
//        Pt{  0,  0 },
//        Pt{ 10,  0 },
//        Pt{  0, 10 },
//    } );
//    ConvexHull<2> dh( {
//        Pt{  7, -2 },
//        Pt{  7,  7 },
//        Pt{ -2,  7 },
//    } );
//    ConvexHull<2> ih = ch.intersection( dh );
//    P( ih );

//    VtkOutput vo;
//    ih.display_vtk( vo );
//    vo.save( "out.vtk" );
//}

//void test_intersection_3D() {
//    using Pt = ConvexHull<3>::Pt;
//    ConvexHull<3> ch( {
//        Pt{  0,  0,  0 },
//        Pt{ 10,  0,  0 },
//        Pt{  0, 10,  0 },
//        Pt{  0,  0, 10 },
//    } );
//    ConvexHull<3> dh( {
//         Pt{  7,  7,  7 },
//         Pt{  5, -2, -2 },
//         Pt{ -2,  5, -2 },
//         Pt{ -2, -2,  5 },
//    } );

//    ConvexHull<3> ih = ch.intersection( dh );
//    P( ih );

//    VtkOutput vo;
//    ch.display_vtk( vo, Pt(  0, 0, 0 ) );
//    dh.display_vtk( vo, Pt(  0, 0, 0 ) );
//    ih.display_vtk( vo, Pt( 10, 0, 0 ) );
//    vo.save( "out.vtk" );
//}

void test_perm() {
    using Ch = ConvexHull<2>;
    using Pt = Ch::Pt;
    using TI = Ch::TI;

    std::vector<TI> perm( 4 );
    std::vector<Pt> pts_a{ Pt{ 0, 0 }, Pt{ 1, 0 }, Pt{ 1, 1 }, Pt{ 0, 1 } };
    std::vector<Pt> pts_b{ Pt{ 0, 0 }, Pt{ 0, 1 }, Pt{ 1, 1 }, Pt{ 1, 0 } };

    Ch a( pts_a );
    Ch b( pts_b );
    P( a.is_a_permutation_of( b, perm.data() ) );
    P( perm );
}

int main() {
    //    test_1D();
    //    test_2D();
    //    test_3D_Wedge();
    //    test_3D_Pyramid();
    //    test_cut();
    //    test_intersection_2D();
    //    test_intersection_3D();
    test_perm();
}
