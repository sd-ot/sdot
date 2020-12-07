#include <parex/containers/CudaAllocator.h>
#include <parex/containers/gtensor.h>
#include <parex/data/TypeInfo.h>
#include <parex/tasks/Task.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "TypeFactory", "[TypeFactory]" ) {
    SECTION( "known type" ) {
        Type *it = Task::type_factory( parex::TypeInfo<int>::name() );
        CHECK( it->name == "parex::SI32" );
    }

    SECTION( "known template" ) {
        Type *gt = Task::type_factory( "parex::gtensor<parex::SI32,3,parex::CudaAllocator>" );
        REQUIRE( gt->parameters.size() == 3 );
        REQUIRE( gt->sub_types.size() == 2 );
        CHECK( gt->parameters[ 1 ] == "3" );
        CHECK( gt->sub_types[ 0 ]->name == "parex::SI32" );
        CHECK( gt->sub_types[ 1 ]->name == "parex::CudaAllocator" );
        CHECK( gt->compilation_environment.includes.contains( "parex/containers/CudaAllocator.h>" ) );
    }

    SECTION( "unknown type" ) {
        Type *ut = Task::type_factory( "Siyc" );
        CHECK( ut->name == "Siyc" );
    }

    SECTION( "unknown template" ) {
        Type *st = Task::type_factory( "Smurf<98,5>" );
        CHECK( st->parameters == std::vector<std::string>{ "98", "5" } );
        CHECK( st->base_name == "Smurf" );
        CHECK( st->name == "Smurf<98,5>" );
    }
}
