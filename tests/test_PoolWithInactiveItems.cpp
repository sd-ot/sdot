#include "../src/sdot/system/PoolWithInactiveItems.h"
#include "../src/sdot/system/Stream.h"
#include "catch_main.h"

struct T {
    int val;
    T  *prev_in_pool;
};

TEST_CASE( "PoolWithInactiveItems", "part_case" ) {
    PoolWithInactiveItems<T> pool;
    T *a = pool.get_item();
    T *b = pool.get_item();
    a->val = 1;
    b->val = 2;
    P( a->val );
    P( b->val );

    pool.free( a );
    a = pool.get_item();
    P( a->val );
}



