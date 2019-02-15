#include "../src/sdot/system/PoolWithInactiveItems.h"
#include "../src/sdot/system/Stream.h"
#include "catch_main.h"

struct T {
    int val;
    T  *prev_in_pool;
    T  *next_in_pool;
};

TEST_CASE( "PoolWithInactiveItems", "part_case" ) {
    PoolWithInactiveItems<T> pool;
    T *a = pool.get_item();
    T *b = pool.get_item();
    a->val = 1;
    b->val = 2;
    CHECK( a != b );
    CHECK( a->val == 1 );
    CHECK( b->val == 2 );

    pool.free( a );
    T *c = pool.get_item();
    CHECK( c == a );
}



