#include <parex/resources/default_CpuAllocator.h>
#include <parex/containers/gvector.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "gtensor", "[containers]" ) {
    SECTION( "from data" ) {
        double data[] = { 1, 2, 3, 4 };
        gvector<double,BasicCpuAllocator> t( &default_CpuAllocator, 4, data, /*own*/ false );
        CHECK( same_repr( t.size(), "4" ) );
        CHECK( same_repr( t, "1 2 3 4" ) );
    }

    SECTION( "from init list" ) {
        gvector<double,BasicCpuAllocator> t( &default_CpuAllocator, { 1, 2, 3, 4 } );
        CHECK( same_repr( t.size(), "4" ) );
        CHECK( same_repr( t, "1 2 3 4" ) );
    }
}
