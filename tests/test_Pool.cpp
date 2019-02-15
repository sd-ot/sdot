#include "../src/sdot/system/Stream.h"
#include "../src/sdot/system/Pool.h"
#include "catch_main.h"

TEST_CASE( "pool", "." ) {
    Pool<int> p;
    int *a = p.New( 10 );
    int *b = p.New( 20 );
    P( a );
    P( b );
    P( *a );
    P( *b );
}



