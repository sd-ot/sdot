#include <parex/wrappers/Scalar.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "Scalar", "[wrapper]" ) {
    CHECK( same_repr( Scalar( 17   )    , 17 ) );
    CHECK( same_repr( Scalar( 17   ) + 2, 19 ) );
    CHECK( same_repr( Scalar( 17   ) * 2, 34 ) );
    CHECK( same_repr( Scalar( 17.5 ) * 2, 35 ) );

    // copy, self op
    Scalar a = 5, b = a;
    b += 1;

    CHECK( same_repr( a, 5 ) );
    CHECK( same_repr( b, 6 ) );
}
