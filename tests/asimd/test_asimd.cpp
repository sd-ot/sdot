#include <asimd/SimdRange.h>
#include <asimd/SimdVec.h>
#include "catch_main.h"
#include "P.h"

template<class V>
bool equal( const V &a, const V &b ) {
    if ( a.size() != b.size() )
        return false;
    for( std::size_t i = 0; i < a.size(); ++i )
        if ( a[ i ] != b[ i ] )
            return false;
    return true;
}

TEST_CASE( "InstructionSet", "[asimd]" ) {
    using namespace asimd::InstructionSet;

    using Is = X86<8,Features::SSE2,Features::AVX2>;
    SECTION( Is::name() ) {
        CHECK( Is::Has<Features::AVX512>::value == 0 );
        CHECK( Is::Has<Features::SSE2  >::value == 1 );
        CHECK( Is::Has<Features::AVX2  >::value == 1 );

        CHECK( Is::SimdSize<std::string>::value == 1 );
        CHECK( Is::SimdSize<double     >::value == 4 );
        CHECK( Is::SimdSize<float      >::value == 8 );
    }
}

TEST_CASE( "SimdVec", "[asimd]" ) {
    using namespace asimd;

    using Is = InstructionSet::X86<8,InstructionSet::Features::SSE2>;
    SECTION( Is::name() ) {
        using VI = SimdVec<int,SimdSize<int>::value>;
        VI v( 10 ), w = VI::iota();

        CHECK( equal( v + w, { 10, 11, 12, 13 } ) );
        CHECK( equal( v - w, { 10,  9,  8,  7 } ) );
        CHECK( equal( v * w, {  0, 10, 20, 30 } ) );
    }
}
TEST_CASE( "SimdRange", "[asimd]" ) {
    using namespace asimd;

    using Is = InstructionSet::X86<8,InstructionSet::Features::SSE2>;
    SECTION( Is::name() ) {
        std::vector<int> simd_sizes, indices;
        SimdRange<SimdSize<int,Is>::value,2>::for_each( 0, 11, [&]( unsigned index, auto simd_size ) {
            simd_sizes.push_back( simd_size );
            indices.push_back( index );
        } );
        CHECK( equal( simd_sizes, { 4, 4, 2,  1 } ) );
        CHECK( equal( indices   , { 0, 4, 8, 10 } ) );
    }
}
