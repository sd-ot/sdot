#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Display/VtkOutput.h"
#include "./catch_main.h"

// #include <boost/multiprecision/mpfr.hpp>

//// nsmake cpp_flag -march=native
//// nsmake lib_name gmp
//// nsmake lib_name mpfr

using namespace sdot;

TEST_CASE( "diam", "diam" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 0 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  Pt = LC::Pt;
    using  std::abs;

    LC cs( LC::Box{ { -10, -10 }, { +10, +10 } } );
    for( TF y : { -1, 0, 1 } )
        for( TF x : { -1, 0, 1 } )
            if ( x || y )
                cs.plane_cut( normalized( Pt( { x, y } ) ), normalized( Pt( { x, y } ) ) );

    CHECK_THAT( cs.boundary_measure(), WithinAbs<TF>( 6.627416998, 1e-4 ) );
    CHECK_THAT( cs.measure         (), WithinAbs<TF>( 3.313708499, 1e-4 ) );

    VtkOutput<1,TF> vo( { "num" } );
    cs.display( vo, { 0 }, 0 );
    cs.display( vo, { 0 }, 1 );

    double off = 0;
    for( Pt n : { Pt{ 1, 0 }, Pt{ -1, 0 }, Pt{ 0, 1 }, Pt{ 0, -1 }, Pt{ 2, 1 } } ) {
        LC ct = cs;
        ct.plane_cut( { 0, 0 }, normalized( n ) );
        off += 2.5;
        for( auto &x : ct.points[ 0 ] )
            x += off;
        ct.display( vo, { off }, 1 );
    }

    //    vo.save( "lc.vtk" );
}

TEST_CASE( "only_sphere", "[!benchmark]" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;

    LC cs( LC::Box{ { -10, -10 }, { +10, +10 } } );
    cs.ball_cut( { 1, 0 }, 1 );

    CHECK_THAT( cs.boundary_measure(), WithinAbs( 2 * M_PI, 1e-6 ) );
    CHECK_THAT( cs.measure         (), WithinAbs( M_PI    , 1e-6 ) );

    VtkOutput<1> vo( { "num" } );
    cs.display( vo, { 0 }, 0 );
    cs.display( vo, { 0 }, 1 );
    //    vo.save( "bc.vtk" );
}


void test_and_display( VtkOutput<1> &vo, std::vector<std::pair<Point2<double>,Point2<double>>> on, int r, int n, unsigned nb_points, double perimeter ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;

    Point2<Pc::TF> off{ r % n * 2.5, - r / n * 2.5 };
    LC icp( LC::Box{ off - Pc::TF( 10 ), off + Pc::TF( 10 ) } );
    for( auto p : on ) {
        icp.plane_cut( off + p.first, normalized( p.second ) );
        vo.add_arrow( { off.x + p.first.x, off.y + p.first.y, 0 }, 0.3 * normalized( Point3<double>{ p.second.x, p.second.y, 0 } ) );
    }
    icp.ball_cut( { off }, 1 );
    icp.display( vo, { Pc::TF( r ) }, 0 );
    icp.display( vo, { Pc::TF( r ) }, 1 );
    CHECK( icp.nb_points() == nb_points );
    CHECK_THAT( icp.boundary_measure(), WithinAbs( perimeter, 1e-4 ) );
}

TEST_CASE( "cases_1" ) {
    using TF = double;
    using Pt = Point2<TF>;

    VtkOutput<1> vo( { "num" } );
    int cpt = 0;

    test_and_display( vo, {
                          { Pt{ +std::sqrt( 0.5 ), 0.0 }, Pt{ +1, 0 } },
                          { Pt{ +std::sqrt( 0.6 ), 0.1 }, Pt{ +1, 0 } }
                      }, cpt++, 3, 2, 3 * M_PI / 2 + std::sqrt( 2 ) );

    test_and_display( vo, {
                          { {  std::sqrt( 0.5 ), 0.0 }, { +1, 0 } },
                          { {               0.0, 0.0 }, { 0, +1 } }
                      }, cpt++, 3, 3, 4.77041 );

    test_and_display( vo, {
                          { { std::sqrt( 0.5 ), 0.0 }, { +1, 0 } },
                          { {              0.0, 0.0 }, { 0, -1 } }
                      }, cpt++, 3, 3, 4.77041 );

    test_and_display( vo, {
                          { {  std::sqrt( 0.5 ), 0.0 }, { +1, 0 } },
                          { {  std::sqrt( 0.2 ), 0.1 }, { +1, 0 } }
                      }, cpt++, 3, 2, 5.85774 );

    test_and_display( vo, {
                          { {  std::sqrt( 0.5 ), 0.0 }, { +1, 0 } },
                          { {  std::sqrt( 0.6 ), 0.1 }, { -1, 0 } }
                      }, cpt++, 3, 0, 0 );

    test_and_display( vo, {
                          { { +.7, 0.0 }, { +1, 0 } },
                          { { 0.0, +.7 }, { 0, +1 } },
                          { { -.7, 0.0 }, { -1, 0 } },
                          { { 0.0, -.7 }, { 0, -1 } }
                      }, cpt++, 3, 4, 5.6 );

    vo.save( "cut2.vtk" );
}

TEST_CASE( "known_values" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  Pt = LC::Pt;

    LC icp( LC::Box{ { -10, -10 }, { +10, +10 } } );
    SECTION( "full" ) {
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI, 1e-7 ) );
    }

    SECTION( "no modification" ) {
        icp.plane_cut( { 1.0, 0.0 }, { 1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 0 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI, 1e-7 ) );
    }

    SECTION( "full removal" ) {
        icp.plane_cut( { 1.0, 0.0 }, { -1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 0 );
        CHECK_THAT( icp.measure(), WithinAbs( 0, 1e-7 ) );
    }

    SECTION( "1 intersection, at the middle" ) {
        for( size_t i = 0, n = 12; i < n; ++i ) {
            icp = LC::Box{ { -10, -10 }, { +10, +10 } };
            double a = 2 * M_PI * i / n;
            icp.plane_cut( { 0.0, 0.0 }, { cos( a ), sin( a ) } );
            icp.ball_cut( { 0.0, 0.0 }, 1 );
            CHECK( icp.nb_points() == 2 );
            CHECK_THAT( icp.measure(), WithinAbs( M_PI / 2, 1e-7 ) );
        }
    }

    SECTION( "1 intersection at the right" ) {
        icp.plane_cut( { std::sqrt( 0.5 ), 0.0 }, { 1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 2 );
        CHECK_THAT( icp.measure(), WithinAbs( 3 * M_PI / 4 + 0.5, 1e-7 ) );
    }

    SECTION( "1 intersection at the left" ) {
        icp.plane_cut( { - std::sqrt( 0.5 ), 0.0 }, { 1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 2 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 4 - 0.5, 1e-7 ) );
    }

    SECTION( "2 intersections of the circle" ) {
        icp.plane_cut( { - std::sqrt( 0.5 ), 0.0 }, { -1, 0 } );
        icp.plane_cut( { + std::sqrt( 0.5 ), 0.0 }, { +1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 4 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 2 + 1.0, 1e-7 ) );
    }

    SECTION( "2 intersections of the circle, the second canceling the first one" ) {
        icp.plane_cut( { - std::sqrt( 0.5 ), 0.0 }, { -1, 0 } );
        icp.plane_cut( { + std::sqrt( 0.5 ), 0.0 }, { -1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 2 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 4 - 0.5, 1e-7 ) );
    }

    SECTION( "2 intersections of the circle, one being worthless" ) {
        icp.plane_cut( { - std::sqrt( 0.5 ), 0.0 }, { -1, 0 } );
        icp.plane_cut( { - 0.9             , 0.0 }, { -1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 2 );
        CHECK_THAT( icp.measure(), WithinAbs( 3 * M_PI / 4 + 0.5, 1e-7 ) );
    }

    SECTION( "2 intersections, one of the line, one of the circle" ) {
        icp.plane_cut( { 0.0, 0.0 }, { -1, 0 } );
        icp.plane_cut( { 0.0, 0.0 }, { 0, +1 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 3 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 4, 1e-7 ) );
    }

    SECTION( "." ) {
        icp.plane_cut( { 0.0, 0.0 }, { -1, 0 } );
        icp.plane_cut( { 0.0, 0.0 }, { 0, -1 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 3 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 4, 1e-7 ) );
    }

    SECTION( "only lines at the end" ) {
        icp.plane_cut( { +0.2,  0.0 }, { +1, 0 } );
        icp.plane_cut( {  0.0, +0.2 }, { 0, +1 } );
        icp.plane_cut( { -0.2,  0.0 }, { -1, 0 } );
        icp.plane_cut( {  0.0, -0.2 }, { 0, -1 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK( icp.nb_points() == 4 );
        CHECK_THAT( icp.measure(), WithinAbs( 0.16, 1e-7 ) );
    }

    SECTION( "degeneracies" ) {
        icp.plane_cut( { 0.0, -1.0 }, { +1, 0 } );
        icp.plane_cut( { 0.0, -1.0 }, normalized( Pt{ 1, 1 } ) );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK_THAT( icp.measure(), WithinAbs( M_PI / 4 - 0.5, 1e-7 ) );
    }
}

TEST_CASE( "centroid" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  Pt = LC::Pt;

    LC icp( LC::Box{ { -10, -10 }, { +10, +10 } } );

    SECTION( "full ball" ) {
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK_THAT( icp.centroid().x, WithinAbs( 0.0, 1e-6 ) );
        CHECK_THAT( icp.centroid().y, WithinAbs( 0.0, 1e-6 ) );
    }

    SECTION( "half disc" ) {
        icp.plane_cut( { 0.0, 0.0 }, { 1, 0 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK_THAT( -0.424413, WithinAbs( icp.centroid().x, 1e-4 ) );
        CHECK_THAT(  0.0     , WithinAbs( icp.centroid().y, 1e-4 ) );
    }

    SECTION( "quarter of a disc" ) {
        icp.plane_cut( { 0.0, 0.0 }, { 1, 0 } );
        icp.plane_cut( { 0.0, 0.0 }, { 0, 1 } );
        icp.ball_cut( { 0.0, 0.0 }, 1 );
        CHECK_THAT( -0.424413, WithinAbs( icp.centroid().x, 1e-4 ) );
        CHECK_THAT( -0.424413, WithinAbs( icp.centroid().y, 1e-4 ) );
    }

    SECTION( "quarter of a disc, radius = 0.5" ) {
        icp.plane_cut( { 0.0, 0.0 }, { 1, 0 } );
        icp.plane_cut( { 0.0, 0.0 }, { 0, 1 } );
        icp.ball_cut( { 0.0, 0.0 }, 0.5 );
        CHECK_THAT( -0.212207, WithinAbs( icp.centroid().x, 1e-4 ) );
        CHECK_THAT( -0.212207, WithinAbs( icp.centroid().y, 1e-4 ) );
    }

    SECTION( "only lines" ) {
        icp.plane_cut( { +1,  0 }, { +1,  0 } );
        icp.plane_cut( {  0, +1 }, {  0, +1 } );
        icp.plane_cut( { -1,  0 }, { -1,  0 } );
        icp.plane_cut( {  0, -1 }, {  0, -1 } );
        icp.ball_cut( { 0.0, 0.0 }, 10 );
        CHECK_THAT( icp.centroid().x, WithinAbs( 0.0, 1e-6 ) );
        CHECK_THAT( icp.centroid().y, WithinAbs( 0.0, 1e-6 ) );
    }
}

TEST_CASE( "integration_only_lines" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; }; // boost::multiprecision::mpfr_float_100
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  Pt = LC::Pt;

    LC icp( LC::Box{ { 0, 0 }, { 2, 1 } } );
    icp.sphere_center = { 0, 0 };
 
    // Unit
    CHECK_THAT( 2.0            , WithinAbs( icp.integration( FunctionEnum::Unit    () )     , 1e-5 ) );
    CHECK_THAT( 1.0            , WithinAbs( icp.centroid   ( FunctionEnum::Unit    () )[ 0 ], 1e-5 ) );
    CHECK_THAT( 0.5            , WithinAbs( icp.centroid   ( FunctionEnum::Unit    () )[ 1 ], 1e-5 ) );
 
    // Gaussian. With wolfram alpha: N[ Integrate[ Integrate[ Exp[ - x*x - y*y ], { x, 0, 2 } ], { y, 0, 1 } ] ]
    // N[ Integrate[ Integrate[ x * Exp[ - x*x - y*y ], { x, 0, 2 } ], { y, 0, 1 } ] ] / N[ Integrate[ Integrate[ Exp[ - x*x - y*y ], { x, 0, 2 } ], { y, 0, 1 } ] ]
    CHECK_THAT( 0.6587596697261, WithinAbs( icp.integration( FunctionEnum::ExpWmR2db<TF>{ 1 } )     , 1e-5 ) );
    CHECK_THAT( 0.5564590588759, WithinAbs( icp.centroid   ( FunctionEnum::ExpWmR2db<TF>{ 1 } )[ 0 ], 1e-5 ) );
    CHECK_THAT( 0.423206       , WithinAbs( icp.centroid   ( FunctionEnum::ExpWmR2db<TF>{ 1 } )[ 1 ], 1e-5 ) );
 
    // r^2. With wolfram alpha: Integrate[ Integrate[ x * ( x * x + y * y ), { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ x * x + y * y, { x, 0, 2 } ], { y, 0, 1 } ]
    CHECK_THAT( 10.0 / 3.0     , WithinAbs( icp.integration( FunctionEnum::R2      () )     , 1e-5 ) );
    // CHECK_THAT( 1.4         , WithinAbs( icp.centroid   ( FunctionEnum::R2      () )[ 0 ], 1e-5 ) );
    // CHECK_THAT( 0.55        , WithinAbs( icp.centroid   ( FunctionEnum::R2      () )[ 1 ], 1e-5 ) );

    // r^4. With wolfram alpha: Integrate[ Integrate[ ( x * x + y * y ) ^ 2, { x, 0, 2 } ], { y, 0, 1 } ]
    CHECK_THAT( 386.0 / 45.0     , WithinAbs( icp.integration( FunctionEnum::R4      () )     , 1e-5 ) );
}

//TEST_CASE( PowerDiagram::ConvexPolyhedron2, integration_line_and_disc ) {
//    PowerDiagram::ConvexPolyhedron2<double> icp( { 0, 0 }, 100 );
//    icp.plane_cut( { 0, 0 }, { -1, 0 } );
//    icp.plane_cut( { 0, 0 }, { 0, -1 } );
//    icp.plane_cut( { 2, 1 }, { +1, 0 } );
//    icp.plane_cut( { 2, 1 }, { 0, +1 } );
//    icp.ball_cut( { 0, 0 }, 1.5 );

//    // Unit
//    CHECK_THAT( 1.37996 , icp.integration( FunctionEnum::Unit    () ), 1e-5 );
//    CHECK_THAT( 0.694464, icp.centroid   ( FunctionEnum::Unit    () )[ 0 ], 1e-5 );
//    CHECK_THAT( 0.47766 , icp.centroid   ( FunctionEnum::Unit    () )[ 1 ], 1e-5 );

//    // Gaussian
//    CHECK_THAT( 0.629767, icp.integration( FunctionEnum::Gaussian() ), 1e-5 );
//    CHECK_THAT( 0.509256, icp.centroid   ( FunctionEnum::Gaussian() )[ 0 ], 1e-5 );
//    CHECK_THAT( 0.418426, icp.centroid   ( FunctionEnum::Gaussian() )[ 1 ], 1e-5 );

//    // r^2
//    CHECK_THAT( 1.31953 , icp.integration( FunctionEnum::R2      () ), 1e-5 );
//    CHECK_THAT( 0.921255, icp.centroid   ( FunctionEnum::R2      () )[ 0 ], 1e-5 );
//    CHECK_THAT( 0.533156, icp.centroid   ( FunctionEnum::R2      () )[ 1 ], 1e-5 );
//}


//TEST_CASE( PowerDiagram::ConvexPolyhedron2, integration_only_disc ) {
//    PowerDiagram::ConvexPolyhedron2<double> icp( { 0, 0 }, 100 );
//    icp.ball_cut( { 0, 0 }, 1.5 );

//    CHECK_THAT( M_PI * 1.5 * 1.5, icp.integration( FunctionEnum::Unit    () ), 1e-5 );
//    CHECK_THAT( 2.81047         , icp.integration( FunctionEnum::Gaussian() ), 1e-5 );
//    CHECK_THAT( 7.95216         , icp.integration( FunctionEnum::R2         () ), 1e-5 );
//}



