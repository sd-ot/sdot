#include <parex/wrappers/String.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "String", "[wrapper]" ) {
    CHECK( same_repr( String( std::string( "cdpso" ) ), "cdpso" ) );
    CHECK( same_repr( String( "cdpso" ), "cdpso" ) );
    CHECK( same_repr( String( "cdpso" ).size(), 5 ) );
    CHECK( same_repr( String( "c" ) + "s", "cs" ) );
}
