#include <parex/containers/gvector.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "gtensor", "[containers]" ) {
    CpuAllocator allocator;
    double data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    gvector<double,CpuAllocator> t( &allocator, 4, data, /*own*/ false );
    CHECK( same_repr( t.size(), "4" ) );
    CHECK( same_repr( t, "1 2 3 4" ) );
}
