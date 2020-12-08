#include <parex/resources/BasicCudaAllocator.h>
#include <parex/containers/gtensor.h>
#include <parex/resources/HwGraph.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

TEST_CASE( "gtensor", "[containers]" ) {
    BasicCpuAllocator allocator;

    SECTION( "from data" ) {
        double data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        gtensor<double,2,BasicCpuAllocator> t( &allocator, { 2, 4 }, data, /*own*/ false );
        CHECK( same_repr( t.shape(), "2,4" ) );
        CHECK( same_repr( t, "1 2 3 4\n5 6 7 8" ) );
    }

    SECTION( "from init list, order = 1" ) {
        gtensor<double,1,BasicCpuAllocator> v( &allocator, { 0, 1, 2 } );
        CHECK( same_repr( v, "0 1 2" ) );
    }

    SECTION( "from init list, order = 2" ) {
        gtensor<double,2,BasicCpuAllocator> u( &allocator, { { 0, 1, 2 }, { 3, 4, 5 } } );
        CHECK( same_repr( u, "0 1 2\n3 4 5" ) );

        u.for_each_offset_and_index( [&]( std::size_t o, std::size_t i, std::size_t j ) {
            CHECK( u.offset( i, j ) == o );
        } );
    }

    #if __has_include(<cuda_runtime.h>)
    if ( default_hw_graph()->nb_cuda_devices() ) {
        SECTION( "gpu tensor, from cpu init list" ) {
            BasicCudaAllocator cuda_allocator{ 0 };
            gtensor<double,2,BasicCudaAllocator> u( &cuda_allocator, { { 0, 1, 2 }, { 3, 4, 5 } } );
            CHECK( same_repr( u, "0 1 2\n3 4 5" ) );
        }
    }
    #endif
}
