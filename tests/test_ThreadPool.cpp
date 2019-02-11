#include "../src/PowerDiagram/system/ThreadPool.h"
#include "../src/PowerDiagram/system/Stream.h"
#include <gtest/gtest.h>

TEST( ThreadPool, part_case ) {
    constexpr int nb_threads = 2;
    constexpr int nb_jobs = 4;

    ThreadPool tp;
    tp.init( nb_threads );
    std::vector<int> done( nb_jobs, -1 );
    tp.execute( 4, [&]( int job, int nth ) {
        done[ job ] = nth;
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    } );

    for( std::size_t i = 0; i < nb_jobs; i += nb_threads )
        std::sort( done.begin() + i, done.begin() + i + nb_threads );
    EXPECT_EQ( to_string( done ), "0 1 0 1" );
}



