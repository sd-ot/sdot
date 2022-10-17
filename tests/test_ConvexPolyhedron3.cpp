#include "../src/sdot/Geometry/ConvexPolyhedron3.h"
#include "catch_main.h"

struct Pc { using TF = double; using TI = std::size_t; using CI = int; enum { allow_ball_cut = false }; };
using  Cp = sdot::ConvexPolyhedron3<Pc>;
using  Pt = sdot::Point3<Pc::TF>;
using  TF = Pc::TF;

using std::abs;

TEST_CASE( "ConvexPolyhedron3 diam" ) {
   Cp cs( Cp::Box{ { -5, -5, -5 }, { +5, +5, +5 } } );
   for( double z : { -1, 0, 1 } )
       for( double y : { -1, 0, 1 } )
           for( double x : { -1, 0, 1 } )
               if ( x || y || z )
                   cs.plane_cut( normalized( Pt( { x, y, z } ) ), normalized( Pt( { x, y, z } ) ) );

   CHECK( abs( cs.boundary_measure() - 14.3319 ) < 1e-4 );
   CHECK( abs( cs.measure         () - 4.77730 ) < 1e-4 );

   CHECK( cs.contains( { 0, 0, 0 } ) == 1 );
   CHECK( cs.contains( { 2, 0, 0 } ) == 0 );

   sdot::VtkOutput<1> vo( { "num" } );
   cs.display( vo, { 0 } );
   vo.save( "cut.vtk" );
}

//TEST( LaguerreCell_3D, only_sphere ) {
//    LaguerreCell<TF,std::string> cs( { 0, 0, 0 }, 100 );
//    cs.sphere_cut( { 0, 0, 0 }, 1 );

//    EXPECT_NEAR( 4.0 / 3.0 * M_PI, cs.volume(), 1e-6 );
//    EXPECT_NEAR( 4.0       * M_PI, cs.area  (), 1e-6 );

//    VtkOutput<1> vo( { "num" } );
//    cs.display( vo, { 0 } );
//    vo.save( "sphere_cut.vtk" );
//}


//// start with a cube on +/- 1, then cut it with a sphere defined by (center, radius)
//void test_sphere_cut( VtkOutput<1> &vo, int cpt, PO center, TF radius, TF expected_area, TF expected_volume, PO expected_centroid ) {
//    PO o = { 2.5 * ( cpt % 3 ), 2.5 * ( cpt / 3 ), 0 };

//    LaguerreCell<TF,std::string> cs( o, 100 );
//    cs.plane_cut( o + PO{ -1, -1, -1 }, { -1, 0, 0 } );
//    cs.plane_cut( o + PO{ -1, -1, -1 }, { 0, -1, 0 } );
//    cs.plane_cut( o + PO{ -1, -1, -1 }, { 0, 0, -1 } );
//    cs.plane_cut( o + PO{ +1, +1, +1 }, { +1, 0, 0 } );
//    cs.plane_cut( o + PO{ +1, +1, +1 }, { 0, +1, 0 } );
//    cs.plane_cut( o + PO{ +1, +1, +1 }, { 0, 0, +1 } );

//    cs.sphere_cut( o + center, radius );

//    EXPECT_NEAR( expected_area      , cs.area()  , 1e-3 );
//    EXPECT_NEAR( expected_volume    , cs.volume(), 1e-3 );

//    if ( cs.volume() ) {
//        PO c = cs.centroid() - o;
//        EXPECT_NEAR( expected_centroid.x, c.x    , 1e-3 );
//        EXPECT_NEAR( expected_centroid.y, c.y    , 1e-3 );
//        EXPECT_NEAR( expected_centroid.z, c.z    , 1e-3 );
//    }

//    cs.display( vo, { 0 }, 0, 1e-1 );
//    cs.display( vo, { 0 }, 1, 1e-1 );
//}

//TEST( LaguerreCell_3D, sphere ) {
//    VtkOutput<1> vo( { "num" } );
//    int cpt = 0;

//    test_sphere_cut( vo, cpt++, { 0, 0, 0 }, 0.9, 10.1788 , 3.05363     , { 0       , 0       , 0 } ); // sphere is fully inside the cube
//    test_sphere_cut( vo, cpt++, { 0, 0, 0 }, 2.0, 24      , 8           , { 0       , 0       , 0 } ); // sphere fully contains the cube
//    test_sphere_cut( vo, cpt++, { 3, 0, 0 }, 3.0, 14.3078 , 3.54326     , { 0.554144, 0       , 0 } ); // sphere cuts half of the cube
//    test_sphere_cut( vo, cpt++, { 0, 0, 0 }, 1.5, 23.0189 , 7.90072     , { 0       , 0       , 0 } ); // sphere cuts all the edges
//    test_sphere_cut( vo, cpt++, { 2, 0, 0 }, 1.5, 8.1839  , 1.03162     , { 0.826109, 0       , 0 } ); // sphere cuts the +x edges
//    test_sphere_cut( vo, cpt++, { 0, 0, 0 }, 1.3, 19.5407 , 7.16702     , { 0       , 0       , 0 } ); // sphere cuts all the faces but no edge
//    test_sphere_cut( vo, cpt++, { 1, 1, 0 }, 2.3, 21.6054 , 7.16963     , { 0.079608, 0.079608, 0 } ); // sphere cuts in the middle and in the -x face
//    test_sphere_cut( vo, cpt++, { 1, 0, 0 }, 1  , 3 * M_PI, 2 * M_PI / 3, { 0.625   , 0       , 0 } ); // hemi sphere
//    test_sphere_cut( vo, cpt++, { 5, 0, 0 }, 1  , 0       , 0           , { 0       , 0       , 0 } ); // sphere fully outside

//    vo.save( "cut.vtk" );
//}

//TEST( LaguerreCell_3D, part_case_1 ) {
//    VtkOutput<1> vo( { "num" } );

//    TF l = 5, h = 5;
//    TF m = 1 * l / 4, n = 3 * l / 4;

//    LaguerreCell<TF,std::string> cs( { 0, 0, 0 }, 100 );

//    // box
//    cs.plane_cut( { 0, 0, 0 }, { -1, 0, 0 } );
//    cs.plane_cut( { 0, 0, 0 }, { 0, -1, 0 } );
//    cs.plane_cut( { 0, 0, 0 }, { 0, 0, -1 } );
//    cs.plane_cut( { l, l, h }, { +1, 0, 0 } );
//    cs.plane_cut( { l, l, h }, { 0, +1, 0 } );
//    cs.plane_cut( { l, l, h }, { 0, 0, +1 } );

//    cs.plane_cut( { m, m, 0 }, normalized( PO{ -0.5, 0, -1 } ) );
//    cs.plane_cut( { m, m, 0 }, normalized( PO{ 0, -0.5, -1 } ) );
//    cs.plane_cut( { n, n, 0 }, normalized( PO{ +0.5, 0, -1 } ) );
//    cs.plane_cut( { n, n, 0 }, normalized( PO{ 0, +0.5, -1 } ) );

//    PO P0{ 0.861454, 0.861454, 0.490821 }; TF W0 = 0.321062;
//    PO P1{ 0.43322 , 0.433221, 0.9246   }; TF W1 = 0.279741;

//    // ngs
//    PO V = P1 - P0;
//    TF D = norm_2( V );
//    PO N = 1.0 / D * V;
//    TF x = 0.5 * ( D + ( W0 - W1 ) / D );
//    cs.plane_cut( P0 + x * N, N );

//    cs.sphere_cut( P0, std::sqrt( W0 ) );

//    //
//    EXPECT_NEAR( 0.531519, cs.volume(), 1e-4 );
//    EXPECT_NEAR( 3.47155 , cs.area  (), 1e-4 );

//    cs.display( vo, { 0 }, 0, 1e-1 );
//    cs.display( vo, { 0 }, 1, 1e-1 );
//    vo.save( "part_case_1.vtk" );
//}

//TEST( LaguerreCell_3D, part_case_2 ) {
//    VtkOutput<1> vo( { "num" } );

//    std::vector<PO> p;
//    std::vector<TF> r;
//    p.push_back( { 0.82375, 0.823749, 0.614837 }); r.push_back( 0.542204 );
//    p.push_back( { 1.7025, 1.09624, 0.330037 }); r.push_back( 0.45216 );
//    p.push_back( { 2.51452, 1.1214, 0.321228 }); r.push_back( 0.415647 );
//    p.push_back( { 3.33175, 1.08092, 0.330358 }); r.push_back( 0.419138 );
//    p.push_back( { 4.18763, 0.814481, 0.574137 }); r.push_back( 0.424561 );
//    p.push_back( { 1.09624, 1.7025, 0.330037 }); r.push_back( 0.45216 );
//    p.push_back( { 1.67565, 1.67565, -0.638175 }); r.push_back( 0.285742 );
//    p.push_back( { 2.50124, 1.6731, -0.616841 }); r.push_back( 0.279732 );
//    p.push_back( { 3.32749, 1.67323, -0.630881 }); r.push_back( 0.281618 );
//    p.push_back( { 3.92186, 1.67516, 0.33037 }); r.push_back( 0.414559 );
//    p.push_back( { 1.1214, 2.51452, 0.321229 }); r.push_back( 0.415647 );
//    p.push_back( { 1.6731, 2.50124, -0.61684 }); r.push_back( 0.279732 );
//    p.push_back( { 2.50054, 2.50054, -0.600899 }); r.push_back( 0.272599 );
//    p.push_back( { 3.3283, 2.50031, -0.615158 }); r.push_back( 0.27792 );
//    p.push_back( { 3.88535, 2.50153, 0.319552 }); r.push_back( 0.399955 );
//    p.push_back( { 1.08092, 3.33176, 0.330358 }); r.push_back( 0.419138 );
//    p.push_back( { 1.67323, 3.32749, -0.630881 }); r.push_back( 0.281618 );
//    p.push_back( { 2.50031, 3.3283, -0.615159 }); r.push_back( 0.27792 );
//    p.push_back( { 3.32745, 3.32745, -0.629771 }); r.push_back( 0.280631 );
//    p.push_back( { 3.92159, 3.3283, 0.32926 }); r.push_back( 0.410726 );
//    p.push_back( { 0.814481, 4.18763, 0.574137 }); r.push_back( 0.424561 );
//    p.push_back( { 1.67516, 3.92186, 0.33037 }); r.push_back( 0.414559 );
//    p.push_back( { 2.50153, 3.88534, 0.319551 }); r.push_back( 0.399955 );
//    p.push_back( { 3.3283, 3.92159, 0.329261 }); r.push_back( 0.410726 );
//    p.push_back( { 4.18657, 4.18657, 0.571944 }); r.push_back( 0.418327 );
//    p.push_back( { 0.381356, 0.381356, 0.937899 }); r.push_back( 0.414059 );
//    p.push_back( { 1.57478, 0.378328, 0.805001 }); r.push_back( 0.360307 );
//    p.push_back( { 2.50763, 0.404574, 0.753137 }); r.push_back( 0.323133 );
//    p.push_back( { 3.4553, 0.38285, 0.784329 }); r.push_back( 0.327573 );
//    p.push_back( { 4.61429, 0.385577, 0.892806 }); r.push_back( 0.349962 );
//    p.push_back( { 0.378328, 1.57478, 0.805001 }); r.push_back( 0.360307 );
//    p.push_back( { 1.70106, 1.70106, 0.707486 }); r.push_back( 0.332263 );
//    p.push_back( { 2.50296, 1.74781, 0.52884 }); r.push_back( 0.300245 );
//    p.push_back( { 3.31057, 1.69233, 0.690373 }); r.push_back( 0.320625 );
//    p.push_back( { 4.61756, 1.54734, 0.782426 }); r.push_back( 0.323427 );
//    p.push_back( { 0.404574, 2.50763, 0.753136 }); r.push_back( 0.323133 );
//    p.push_back( { 1.74781, 2.50296, 0.528841 }); r.push_back( 0.300245 );
//    p.push_back( { 2.50117, 2.50117, 0.386735 }); r.push_back( 0.265367 );
//    p.push_back( { 3.25961, 2.50059, 0.526299 }); r.push_back( 0.295871 );
//    p.push_back( { 4.59583, 2.50045, 0.746701 }); r.push_back( 0.309997 );
//    p.push_back( { 0.38285, 3.4553, 0.784328 }); r.push_back( 0.327573 );
//    p.push_back( { 1.69233, 3.31057, 0.690373 }); r.push_back( 0.320625 );
//    p.push_back( { 2.50059, 3.25961, 0.526299 }); r.push_back( 0.295871 );
//    p.push_back( { 3.31044, 3.31044, 0.687581 }); r.push_back( 0.31833 );
//    p.push_back( { 4.61697, 3.45402, 0.780693 }); r.push_back( 0.321053 );
//    p.push_back( { 0.385577, 4.61429, 0.892806 }); r.push_back( 0.349962 );
//    p.push_back( { 1.54734, 4.61756, 0.782427 }); r.push_back( 0.323427 );
//    p.push_back( { 2.50045, 4.59583, 0.746701 }); r.push_back( 0.309998 );
//    p.push_back( { 3.45402, 4.61698, 0.780694 }); r.push_back( 0.321053 );
//    p.push_back( { 4.61474, 4.61474, 0.890019 }); r.push_back( 0.344977 );
//    p.push_back( { 0.523059, 0.523059, 1.63072 }); r.push_back( 0.278887 );
//    p.push_back( { 1.49879, 0.514246, 1.523 }); r.push_back( 0.26469 );
//    p.push_back( { 2.49998, 0.509891, 1.49816 }); r.push_back( 0.259614 );
//    p.push_back( { 3.5007, 0.512381, 1.51191 }); r.push_back( 0.260899 );
//    p.push_back( { 4.48128, 0.518731, 1.60332 }); r.push_back( 0.269077 );
//    p.push_back( { 0.514246, 1.49879, 1.523 }); r.push_back( 0.26469 );
//    p.push_back( { 1.49372, 1.49372, 1.44933 }); r.push_back( 0.256065 );
//    p.push_back( { 2.5, 1.49829, 1.41326 }); r.push_back( 0.251561 );
//    p.push_back( { 3.50569, 1.49429, 1.44614 }); r.push_back( 0.255166 );
//    p.push_back( { 4.48781, 1.49931, 1.51064 }); r.push_back( 0.260473 );
//    p.push_back( { 0.509891, 2.49998, 1.49816 }); r.push_back( 0.259614 );
//    p.push_back( { 1.49829, 2.5, 1.41326 }); r.push_back( 0.251561 );
//    p.push_back( { 2.5, 2.5, 1.40008 }); r.push_back( 0.250013 );
//    p.push_back( { 3.50161, 2.5, 1.41274 }); r.push_back( 0.251417 );
//    p.push_back( { 4.49062, 2.5, 1.4945 }); r.push_back( 0.25839 );
//    p.push_back( { 0.512381, 3.5007, 1.51191 }); r.push_back( 0.260899 );
//    p.push_back( { 1.49429, 3.50569, 1.44614 }); r.push_back( 0.255166 );
//    p.push_back( { 2.5, 3.50161, 1.41274 }); r.push_back( 0.251417 );
//    p.push_back( { 3.5056, 3.5056, 1.44555 }); r.push_back( 0.255 );
//    p.push_back( { 4.48794, 3.50067, 1.50985 }); r.push_back( 0.26023 );
//    p.push_back( { 0.518732, 4.48128, 1.60332 }); r.push_back( 0.269077 );
//    p.push_back( { 1.49931, 4.48781, 1.51064 }); r.push_back( 0.260473 );
//    p.push_back( { 2.5, 4.49062, 1.4945 }); r.push_back( 0.25839 );
//    p.push_back( { 3.50067, 4.48794, 1.50985 }); r.push_back( 0.26023 );
//    p.push_back( { 4.48154, 4.48154, 1.6014 }); r.push_back( 0.268382 );


//    TF l = 5, h = 5;
//    TF m = 1 * l / 4, n = 3 * l / 4;

//    LaguerreCell<TF,std::string> cs( { 0, 0, 0 }, 100 );

//    // box
//    cs.plane_cut( { 0, 0, 0 }, { -1, 0, 0 } );
//    cs.plane_cut( { 0, 0, 0 }, { 0, -1, 0 } );
//    cs.plane_cut( { 0, 0, 0 }, { 0, 0, -1 } );
//    cs.plane_cut( { l, l, h }, { +1, 0, 0 } );
//    cs.plane_cut( { l, l, h }, { 0, +1, 0 } );
//    cs.plane_cut( { l, l, h }, { 0, 0, +1 } );

//    cs.plane_cut( { m, m, 0 }, normalized( PO{ -0.5, 0, -1 } ) );
//    cs.plane_cut( { m, m, 0 }, normalized( PO{ 0, -0.5, -1 } ) );
//    cs.plane_cut( { n, n, 0 }, normalized( PO{ +0.5, 0, -1 } ) );
//    cs.plane_cut( { n, n, 0 }, normalized( PO{ 0, +0.5, -1 } ) );

//    // ngs
//    for( size_t i = 1; i < p.size(); ++i ) {
//        PO V = p[ i ] - p[ 0 ];
//        TF D = norm_2( V );
//        PO N = 1.0 / D * V;
//        TF x = 0.5 * ( D + ( r[ 0 ] - r[ i ] ) / D );
//        cs.plane_cut( p[ 0 ] + x * N, N );
//    }

//    cs.sphere_cut( p[ 0 ], std::sqrt( r[ 0 ] ) );

//    //
//    P( cs.volume_mc(), cs.volume(), 1e-4 );
//    P( cs.area_ap  (), cs.area  (), 1e-4 );

//    cs.display( vo, { 0 }, 0, 1e-1 );
//    cs.display( vo, { 0 }, 1, 1e-1 );
//    vo.save( "part_case_2.vtk" );
//}

//TEST( LaguerreCell_3D, dep_cube ) {
//    VtkOutput<1> vo( { "num" } );

//    LaguerreCell<TF,std::string> cs( { 0, 0, 0 }, 100 );
//    cs.plane_cut( { 0, 0, 0 }, { -1,  0,  0 } );
//    cs.plane_cut( { 0, 0, 0 }, {  0, -1,  0 } );
//    cs.plane_cut( { 0, 0, 0 }, {  0,  0, -1 } );
//    cs.plane_cut( { 1, 1, 1 }, { +1,  0,  0 } );
//    cs.plane_cut( { 1, 1, 1 }, {  0, +1,  0 } );
//    cs.plane_cut( { 1, 1, 0.7 }, {  0,  0, +1 } );

//    cs.sphere_cut( { 0.5, 0.5, 0.2 }, 0.6 );

//    EXPECT_NEAR( 0.318587, cs.centroid().z, 1e-3 );
//    EXPECT_NEAR( 0.586745, cs.volume  ()  , 1e-3 );
//    EXPECT_NEAR( 3.79692 , cs.area    ()  , 1e-3 );

//    cs.display( vo, { 0 }, 0, 1e-1 );
//    cs.display( vo, { 0 }, 1, 1e-1 );
//    vo.save( "part_case.vtk" );
//}

//TEST( LaguerreCell_3D, cut_cube ) {
//    VtkOutput<1> vo( { "num" } );

//    LaguerreCell<TF,std::string> cs( { 0, 0, 0 }, 100 );
//    cs.plane_cut( { 0, 0, 0 }, { -1,  0,  0 } );
//    cs.plane_cut( { 0, 0, 0 }, {  0, -1,  0 } );
//    cs.plane_cut( { 0, 0, 0 }, {  0,  0, -1 } );
//    cs.plane_cut( { 1, 1, 1 }, { +1,  0,  0 } );
//    cs.plane_cut( { 1, 1, 1 }, {  0, +1,  0 } );
//    cs.plane_cut( { 1, 1, 1 }, {  0,  0, +1 } );

//    cs.sphere_cut( { 2, 2, 1 }, 1 );

//    EXPECT_NEAR( 0, cs.volume(), 1e-6 );
//    EXPECT_NEAR( 0, cs.area  (), 1e-6 );

//    cs.display( vo, { 0 }, 0, 1e-1 );
//    cs.display( vo, { 0 }, 1, 1e-1 );
//    vo.save( "cut_cube.vtk" );
//}
