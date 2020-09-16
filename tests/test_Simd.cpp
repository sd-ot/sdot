#include "../src/sdot/support/AlignedAllocator.h"
#include "../src/sdot/support/simd/SimdVec.h"
#include "../src/sdot/support/StaticRange.h"
#include "../src/sdot/support/ASSERT.h"
#include "../src/sdot/support/P.h"
#include <bitset>

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3

template<class T,class I,class Arch,unsigned n>
struct Test {
    enum { alig = SimdAlig<T,Arch>::value };
    using  VF   = SimdVec<T,n,Arch>;
    using  VI   = SimdVec<I,n,Arch>;

    void test_aligned_load() {
        T *p = al.template allocate<T,alig>( n );
        for( unsigned i = 0; i < n; ++i )
            p[ i ] = rand();

        VF a = VF::load_aligned( p );
        for( unsigned i = 0; i < n; ++i )
            ASSERT( a[ i ] == p[ i ] );

        al.free( p, n );
    }

    void test_aligned_load_int8() {
        std::vector<std::int8_t> p( n );
        for( unsigned i = 0; i < n; ++i )
            p[ i ] = rand();

        VF a = VF::load_aligned( p.data() );
        for( unsigned i = 0; i < n; ++i )
            ASSERT( a[ i ] == p[ i ] );
    }

    void test_iota() {
        VF a = VF::iota( 17 );
        for( unsigned i = 0; i < n; ++i )
            ASSERT( a[ i ] == 17 + i );
    }

    template<class Func>
    void test_bin_op( const Func &func ) {
        VF a, b;
        for( unsigned i = 0; i < n; ++i ) {
            a[ i ] = rand();
            b[ i ] = rand();
        }

        VF c = func( a, b );
        for( unsigned i = 0; i < n; ++i ) {
            if ( c[ i ] != func( a[ i ], b[ i ] ) ) {
                P( i, c[ i ], func( a[ i ], b[ i ] ) );
                ASSERT( 0 );
            }
        }
    }

    void test_bin_ops( N<0> /* => int type */ ) {
        test_bin_op( []( auto a, auto b ) { return a + b; } );
        test_bin_op( []( auto a, auto b ) { return a - b; } );
        test_bin_op( []( auto a, auto b ) { return a & b; } );
        test_bin_op( []( auto a, auto b ) { return ( a & 15 ) << ( b & 15 ); } );
    }

    void test_bin_ops( N<1> /* => float type */ ) {
        test_bin_op( []( auto a, auto b ) { return a + b; } );
        test_bin_op( []( auto a, auto b ) { return a - b; } );
        test_bin_op( []( auto a, auto b ) { return a * b; } );
        test_bin_op( []( auto a, auto b ) { return a / b; } );
    }

    void test_gather() {
        std::vector<T> p( 32 );
        for( unsigned i = 0; i < p.size(); ++i )
            p[ i ] = rand();

        VI ind;
        for( unsigned i = 0; i < n; ++i )
            ind[ i ] = rand() % p.size();

        VF a = VF::gather( p.data(), ind );
        for( unsigned i = 0; i < n; ++i )
            ASSERT( a[ i ] == p[ ind[ i ] ] );
    }

    void test_scatter() {
        std::vector<T> p( 32, 0 );
        for( unsigned i = 0; i < p.size(); ++i )
            p[ i ] = rand();

        VI ind;
        for( unsigned i = 0, o = p.size() + rand() % p.size(); i < n; ++i )
            ind[ i ] = ( o - i ) % p.size();

        VF values;
        for( unsigned i = 0; i < n; ++i )
            values[ i ] = rand();

        VF::scatter( p.data(), ind, values );
        for( unsigned i = 0; i < n; ++i )
            ASSERT( p[ ind[ i ] ] == values[ i ] );
    }

    void all_tests() {
        constexpr int is_float = std::is_same<T,float>::value || std::is_same<T,double>::value;

        test_aligned_load();
        test_aligned_load_int8();
        test_iota();
        test_bin_ops( N<is_float>() );
        test_gather();
        test_scatter();
    }

    AlignedAllocator al;
};

int main() {
    using TFS = std::tuple<std::uint64_t,std::int64_t,double,std::uint32_t,std::int32_t,float>;
    using TIS = std::tuple<std::uint64_t,std::int64_t,std::uint32_t,std::int32_t>;
    using ARS = std::tuple<MachineArch::SSE2,MachineArch::AVX2,MachineArch::AVX512>;

    StaticRange<std::tuple_size<TFS>::value>::for_each( [&]( auto nt ) {
        using TF = typename std::tuple_element<nt.value,TFS>::type;
        StaticRange<std::tuple_size<TIS>::value>::for_each( [&]( auto ni ) {
            using TI = typename std::tuple_element<ni.value,TIS>::type;
            StaticRange<std::tuple_size<ARS>::value>::for_each( [&]( auto na ) {
                using Arch = typename std::tuple_element<na.value,ARS>::type;
                StaticRange<4,5>::for_each( [&]( auto ne ) {
                    constexpr int n = 1 << ne.value;
                    Test<TF,TI,Arch,n> test;
                    test.all_tests();
                } );
            } );
        } );
    } );
}

