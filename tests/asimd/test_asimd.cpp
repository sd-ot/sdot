#include <asimd/SimdRange.h>
#include <asimd/SimdVec.h>
#include "catch_main.h"
#include "P.h"

using namespace asimd;

template<class V>
bool equal( const V &a, const V &b ) {
    if ( a.size() != b.size() )
        return false;
    for( std::size_t i = 0; i < a.size(); ++i )
        if ( a[ i ] != b[ i ] )
            return false;
    return true;
}

TEST_CASE( "processing_units", "[asimd]" ) {
    using namespace processing_units;

    using Is = X86<8,features::SSE2,features::AVX2>;
    SECTION( Is::name() ) {
        CHECK( Is::Has<features::AVX512>::value == 0 );
        CHECK( Is::Has<features::SSE2  >::value == 1 );
        CHECK( Is::Has<features::AVX2  >::value == 1 );

        CHECK( Is::SimdSize<std::string>::value == 1 );
        CHECK( Is::SimdSize<double     >::value == 4 );
        CHECK( Is::SimdSize<float      >::value == 8 );
    }

    using Cs = NvidiaGpu<8>;
    SECTION( Cs::name() ) {
        CHECK( Cs::SimdSize<float      >::value == 1 );
    }
}

TEST_CASE( "SimdVec", "[asimd]" ) {
    using Is = processing_units::X86<8,processing_units::features::SSE2>;
    SECTION( Is::name() ) {
        using VI = SimdVec<int,SimdSize<int>::value>;
        VI v( 10 ), w = VI::iota();

        CHECK( equal( v + w, { 10, 11, 12, 13 } ) );
        CHECK( equal( v - w, { 10,  9,  8,  7 } ) );
        CHECK( equal( v * w, {  0, 10, 20, 30 } ) );
    }
}
TEST_CASE( "SimdRange", "[asimd]" ) {
    using Is = processing_units::X86<8,processing_units::features::SSE2>;
    SECTION( Is::name() ) {
        std::vector<int> simd_sizes, indices;
        SimdRange<4,2>::for_each( 1, 15, [&]( unsigned index, auto simd_size ) {
            simd_sizes.push_back( simd_size );
            indices.push_back( index );
        } );
        CHECK( equal( simd_sizes, { 1,2,4,4, 2, 1 } ) );
        CHECK( equal( indices   , { 1,2,4,8,12,14 } ) );
    }
}
