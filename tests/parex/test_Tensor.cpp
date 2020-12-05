#include <parex/wrappers/Tensor.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "Tensor", "[wrapper]" ) {
    CHECK( same_repr( Tensor( {{0,1,2},{3,4,5}} ), "0 1 2\n3 4 5" ) );
}
