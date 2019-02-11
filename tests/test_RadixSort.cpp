#include "../src/sdot/system/RadixSort.h"
#include "../src/sdot/system/Tick.h"
#include <gtest/gtest.h>

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3

template<class TI>
void test_with_type() {
    std::size_t n = 100000000ul;
    std::vector<TI> inp;
    for( std::size_t i = 0; i < n; ++i )
        inp.push_back( rand() );

    std::vector<TI> out( inp.size() );
    tick.start( "radix" );
    N<sizeof(TI)> ns;
    TI *res = radix_sort( out.data(), inp.data(), inp.size(), ns );
    tick.stop( "radix" );
    for( std::size_t i = 1; i < inp.size(); ++i )
        EXPECT_GE( res[ i ], res[ i - 1 ] );
    //    P( std::vector<TI>{ res, res + inp.size() } );

    inp.resize( 0 );
    for( std::size_t i = 0; i < n; ++i )
        inp.push_back( rand() );

    tick.start( "sort" );
    std::sort( inp.begin(), inp.end() );
    tick.stop( "sort" );
}

TEST( RadixSort, dim_eq_2 ) {
    //    test_with_type<std::uint8_t >();
    //    test_with_type<std::uint16_t>();
    test_with_type<std::uint32_t>();
}

