#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Display/VtkOutput.h"
#include "../src/sdot/Support/Stream.h"
#include "./catch_main.h"
#include <iostream>

//// nsmake cpp_flag -march=native

using namespace sdot;

TEST_CASE( "Arf", "Arf" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; };
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  Pt = LC::Pt;

    // box { 0, 0 }, { 2, 1 }
    LC icp( LC::Box{ { 0, 0 }, { 0.2, 0.1 } } );
    icp.sphere_center = { 0, 0 };

    //
    FunctionEnum::Arf arf;
    arf.values = []( double r ) -> double {
        if ( r > 1 )
            return 0;
        using std::pow;
        return ( 1 - r * r );
    };
    arf.stops = { 1.0 };

    // w - r^2.
    // Integrate[ Integrate[ 1 - ( x * x + y * y ), { x, 0, 0.2 } ], { y, 0, 0.1 } ] => 0.0196667
    // Integrate[ Integrate[ x * ( 10 - ( x * x + y * y ) ), { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ 10 - ( x * x + y * y ), { x, 0, 2 } ], { y, 0, 1 } ]
    // CHECK_THAT( icp.integration( arf, 1 )     , WithinAbs( 50.0 /  3.0, 1e-5 ) );
    P( icp.integration( arf, 1 ) );
}
