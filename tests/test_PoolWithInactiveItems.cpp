#include "../src/sdot/system/PoolWithInactiveItems.h"
#include "../src/sdot/system/Stream.h"
#include "catch_main.h"

struct T {
    T  *prev_in_pool;
    T  *next_in_pool;
    int val;
};

TEST_CASE( "PoolWithInactiveItems", "part_case" ) {
    PoolWithInactiveItems<T,true> pool;
    T *a = pool.new_item();
    T *b = pool.new_item();
    a->val = 1;
    b->val = 2;
    CHECK( a != b );
    CHECK( a->val == 1 );
    CHECK( b->val == 2 );

    pool.free( a );
    T *c = pool.new_item();
    CHECK( c == a );

    int cpt = 1;
    for( T &i : pool )
        CHECK( i.val == cpt++ );
}
