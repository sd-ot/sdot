#pragma once

#include "SimdVecAggregate.h"
#include "SimdSize.h"
#include "TODO.h"

#include <x86intrin.h>

namespace sdot {

/**
  Generic simd vec.

  (XSIMD would have been a great choice expected for mask handling).
*/
template<class _TF,int _size=SimdSize<ARCH,_TF>::value> struct SimdVec {
    enum {         size        = _size };
    using          T           = _TF;

    /**/           SimdVec      ( T a, T b, T c, T d ) { values[ 0 ] = a; values[ 1 ] = b; values[ 2 ] = c; values[ 3 ] = d; }
    /**/           SimdVec      ( T a, T b ) { values[ 0 ] = a; values[ 1 ] = b; }
    /**/           SimdVec      ( T value ) { for( int i = 0; i < size; ++i ) values[ i ] = value; }
    /**/           SimdVec      () {}

    static void    store_aligned( T *data, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) data[ i ] = vec.values[ i ]; }
    template       <class GF>
    static SimdVec load_aligned ( const GF *data ) { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = data[ i ]; return res; }

    static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

    SimdVec        operator+    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] + that.values[ i ]; return res; }
    SimdVec        operator-    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] - that.values[ i ]; return res; }
    SimdVec        operator*    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] * that.values[ i ]; return res; }
    SimdVec        operator/    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] / that.values[ i ]; return res; }

    SimdVec        operator<<   ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] << that.values[ i ]; return res; }
    SimdVec        operator&    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] & that.values[ i ]; return res; }

    std::uint64_t  operator>    ( const SimdVec &that ) const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] > that.values[ i ] ) << i; return res;  }
    std::uint64_t  neg          () const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] <  0 ) << i; return res; }
    std::uint64_t  nz           () const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] != 0 ) << i; return res; }

    const T&       operator[]   ( int n ) const { return values[ n ]; }
    T&             operator[]   ( int n ) { return values[ n ]; }
    const T       *begin        () const { return values; }
    const T       *end          () const { return values + size; }

    T              values       [ size ];
};

#define SIMD_AGGREGATE( TF, USED_SIZE, TARGET_SIZE ) \
    template<> struct SimdVec<TF,TARGET_SIZE> : SimdVecAggregate<SimdVec<TF,TARGET_SIZE>,SimdVec<TF,USED_SIZE>,TARGET_SIZE/USED_SIZE> { using SimdVecAggregate<SimdVec<TF,TARGET_SIZE>,SimdVec<TF,USED_SIZE>,TARGET_SIZE/USED_SIZE>::SimdVecAggregate; }

#ifdef __SSE2__
    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<std::uint64_t,2> {
        enum {         size         = 2 };
        using          T            = std::uint64_t;

        /**/           SimdVec      ( __m128i values ) : values( values ) {}
        /**/           SimdVec      ( T a, T b ) { values = _mm_set_epi64x( b, a ); }
        /**/           SimdVec      ( T value ) { values = _mm_set1_epi64x( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm_store_si128( (__m128i *)data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm_load_si128( reinterpret_cast<const __m128i *>( data ) ); }
        // static SimdVec load_aligned ( const std::uint8_t *data ) { return _mm256_cvtepi8_epi64( _mm_set1_epi32( *reinterpret_cast<const std::uint32_t *>( data ) ) ); }
        static SimdVec from_int8s   ( std::uint64_t val ) { return _mm_cvtsi64_si128( val ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        SimdVec        operator<<   ( const SimdVec &that ) const { return _mm_sllv_epi64( values, that.values ); }
        SimdVec        operator&    ( const SimdVec &that ) const { return _mm_and_si128( values, that.values ); }

        std::uint64_t  neg          () const { return _mm_movemask_pd( (__m128d)values ); }
        std::uint64_t  nz           () const { return _mm_movemask_pd( (__m128d)_mm_xor_si128( _mm_set1_epi8( -1 ), _mm_cmpeq_epi64( values, _mm_setzero_si128() ) ) ); }

        // SimdVec     permute      ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m128i        values;
    };

    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<double,2> {
        enum {         size         = 2 };
        using          T            = double;

        /**/           SimdVec      ( __m128d values ) : values( values ) {}
        /**/           SimdVec      ( T a, T b ) { values = _mm_set_pd( b, a ); }
        /**/           SimdVec      ( T value ) { values = _mm_set1_pd( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm_load_pd( data ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { __m128d c = _mm_cmpgt_pd( values, that.values ); return _mm_movemask_pd( c );  }
        std::uint64_t  neg          () const { return _mm_movemask_pd( values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m128d        values;
    };
#endif

#ifdef __AVX2__
    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<std::uint64_t,4> {
        enum {         size         = 4 };
        using          T            = std::uint64_t;

        /**/           SimdVec      ( T a, T b, T c, T d ) : values( _mm256_set_epi64x( d, c, b, a ) ) {}
        /**/           SimdVec      ( __m256i values ) : values( values ) {}
        /**/           SimdVec      ( T value ) { values = _mm256_set1_epi64x( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm256_store_si256( reinterpret_cast<__m256i *>( data ), vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm256_load_si256( reinterpret_cast<const __m256i *>( data ) ); }
        static SimdVec load_aligned ( const std::uint8_t *data ) { return _mm256_cvtepi8_epi64( _mm_set1_epi32( *reinterpret_cast<const std::uint32_t *>( data ) ) ); }
        static SimdVec from_int8s   ( std::uint64_t val ) { return _mm256_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        SimdVec        operator<<   ( const SimdVec &that ) const { return _mm256_sllv_epi64( values, that.values ); }
        SimdVec        operator&    ( const SimdVec &that ) const { return _mm256_and_si256( values, that.values ); }

        std::uint64_t  neg          () const { return _mm256_movemask_pd( (__m256d)values ); }
        std::uint64_t  nz           () const { return _mm256_movemask_pd( (__m256d)_mm256_xor_si256( _mm256_set1_epi8( -1 ), _mm256_cmpeq_epi64( values, _mm256_setzero_si256() ) ) ); }

        // SimdVec     permute      ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m256i        values;
    };

    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<double,4> {
        enum {         size         = 4 };
        using          T            = double;

        /**/           SimdVec      ( __m256d values ) : values( values ) {}
        /**/           SimdVec      ( double a, double b, double c, double d ) : values( _mm256_set_pd( d, c, b, a ) ) {}
        /**/           SimdVec      ( SimdVec<double,2> a, SimdVec<double,2> b ) : values( _mm256_set_m128d( b.values, a.values ) ) {}
        /**/           SimdVec      ( T value ) { values = _mm256_set1_pd( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm256_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm256_load_pd( data ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { __m256d c = _mm256_cmp_pd( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_pd( c );  }
        std::uint64_t  neg          () const { return _mm256_movemask_pd( values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m256d        values;
    };
#endif


#ifdef __AVX512F__
    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<std::uint64_t,8> {
        enum {         size         = 8 };
        using          T            = std::uint64_t;

        /**/           SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { values = _mm512_set_epi64( h, g, f, e, d, c, b, a ); }
        /**/           SimdVec      ( __m512i values ) : values( values ) {}
        /**/           SimdVec      ( T value ) { values = _mm512_set1_epi64( value ); }
        /**/           SimdVec      () {}

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm512_store_epi64( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm512_load_epi64( data ); }
        static SimdVec load_aligned ( const std::uint8_t *data ) { return _mm512_cvtepi8_epi64( _mm_set1_epi64x( *reinterpret_cast<const std::uint64_t *>( data ) ) ); }
        static SimdVec from_int8s   ( std::uint64_t val ) { return _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        SimdVec        operator<<   ( const SimdVec &that ) const { return _mm512_sllv_epi64( values, that.values ); }
        SimdVec        operator&    ( const SimdVec &that ) const { return _mm512_and_epi64( values, that.values ); }

        std::uint64_t  neg          () const { return _mm512_movepi64_mask( values ); }
        std::uint64_t  nz           () const { return _mm512_cmpneq_epi64_mask( values, _mm512_setzero_si512() ); }

        SimdVec        permute      ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_epi64( values, idx.values, b.values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m512i        values;
    };

    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<double,8> {
        enum {         size         = 8 };
        using          T            = double;

        /**/           SimdVec      ( __m512d values ) : values( values ) {}
        /**/           SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { values = _mm512_set_pd( h, g, f, e, d, c, b, a ); }
        /**/           SimdVec      ( T value ) { values = _mm512_set1_pd( value ); }
        /**/           SimdVec      () {}

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm512_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm512_load_pd( data ); }

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { return _mm512_cmp_pd_mask( values, that.values, _CMP_GT_OQ );  }
        std::uint64_t  neg          () const { return _mm512_movepi64_mask( (__m512i)values ); }

        SimdVec        permute      ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_pd( values, idx.values, b.values ); }

        const T&       operator[]   ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
        T&             operator[]   ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m512d        values;
    };
#endif

#if defined(__AVX512F__)
    SIMD_AGGREGATE( double       , 8, 16 );
    SIMD_AGGREGATE( std::uint64_t, 8, 16 );
#elif defined(__AVX2__)
    SIMD_AGGREGATE( double       , 4,  8 );
    SIMD_AGGREGATE( double       , 4, 16 );
    SIMD_AGGREGATE( std::uint64_t, 4,  8 );
    SIMD_AGGREGATE( std::uint64_t, 4, 16 );
#elif defined(__SSE2__)
    SIMD_AGGREGATE( double       , 2,  4 );
    SIMD_AGGREGATE( double       , 2,  8 );
    SIMD_AGGREGATE( double       , 2, 16 );
    SIMD_AGGREGATE( std::uint64_t, 2,  4 );
    SIMD_AGGREGATE( std::uint64_t, 2,  8 );
    SIMD_AGGREGATE( std::uint64_t, 2, 16 );
#endif

} // namespace sdot
