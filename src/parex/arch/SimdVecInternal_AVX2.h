#pragma once

#include "SimdVecInternal.h"
#include <x86intrin.h>
#include "SimdSize.h"

namespace parex {

DECL_SIMD_SIZE( std::uint64_t, Arch::AVX2, 4 );
DECL_SIMD_SIZE( std::int64_t , Arch::AVX2, 4 );
DECL_SIMD_SIZE( double       , Arch::AVX2, 4 );
DECL_SIMD_SIZE( std::uint32_t, Arch::AVX2, 8 );
DECL_SIMD_SIZE( std::int32_t , Arch::AVX2, 8 );
DECL_SIMD_SIZE( float        , Arch::AVX2, 8 );

#ifdef __AVX2__
namespace SimdVecInternal {

// struct Impl<...>
SIMD_VEC_IMPL_REG( Arch::avx2, std::uint64_t, 4, __m256i );
SIMD_VEC_IMPL_REG( Arch::avx2, std::int64_t , 4, __m256i );
SIMD_VEC_IMPL_REG( Arch::avx2, double       , 4, __m256d );
SIMD_VEC_IMPL_REG( Arch::avx2, std::uint32_t, 8, __m256i );
SIMD_VEC_IMPL_REG( Arch::avx2, std::int32_t , 8, __m256i );
SIMD_VEC_IMPL_REG( Arch::avx2, float        , 8, __m256  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, std::uint64_t, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, std::int64_t , 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, double       , 4, _mm256_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, std::uint32_t, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, std::int32_t , 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx, float        , 8, _mm256_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_4( Arch::avx, std::uint64_t, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( Arch::avx, std::int64_t , 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( Arch::avx, double       , 4, _mm256_set_pd( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( Arch::avx, std::uint32_t, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( Arch::avx, std::int32_t , 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( Arch::avx, float        , 8, _mm256_set_ps( h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, std::uint64_t, 4, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, std::int64_t , 4, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, double       , 4, _mm256_load_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, std::uint32_t, 8, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, std::int32_t , 8, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx, float        , 8, _mm256_load_ps( data ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, std::uint64_t, 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, std::int64_t , 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, double       , 4, _mm256_loadu_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, std::uint32_t, 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, std::int32_t , 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( Arch::avx, float        , 8, _mm256_loadu_ps( data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, std::uint64_t, 4, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, std::int64_t , 4, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, double       , 4, _mm256_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, std::uint32_t, 8, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, std::int32_t , 8, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx, float        , 8, _mm256_store_ps   (            data, impl.data.reg ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE( Arch::avx, std::uint64_t, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx, std::int64_t , 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx, double       , 4, _mm256_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx, std::uint32_t, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx, std::int32_t , 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx, float        , 8, _mm256_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, std::uint64_t, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, std::int64_t , 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, double       , 4, NAME, _mm256_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, std::uint32_t, 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, std::int32_t , 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, float        , 8, NAME, _mm256_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( sub );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( min );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, double, 4, NAME, _mm256_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx, float , 8, NAME, _mm256_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint64_t, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int64_t , 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx , double       , 4, anb, _mm256_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint32_t, 8, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int32_t , 8, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx , float        , 8, anb, _mm256_and_ps    );


//// arithmetic operations that work only on int types -------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint64_t, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int64_t , 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint32_t, 8, sll, _mm256_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int32_t , 8, sll, _mm256_sllv_epi32 );

// => SSE2 size
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint64_t, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int64_t , 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::uint32_t, 4, sll, _mm_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx2, std::int32_t , 4, sll, _mm_sllv_epi32 );

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint64_t, std::uint32_t, 4, _mm256_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int64_t , std::uint32_t, 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, double       , std::uint32_t, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint32_t, std::uint32_t, 8, _mm256_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int32_t , std::uint32_t, 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, float        , std::uint32_t, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint64_t, std::int32_t , 4, _mm256_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int64_t , std::int32_t , 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, double       , std::int32_t , 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint32_t, std::int32_t , 8, _mm256_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int32_t , std::int32_t , 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, float        , std::int32_t , 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint64_t, std::uint32_t, 2, _mm_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int64_t , std::uint32_t, 2, _mm_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, double       , std::uint32_t, 2, _mm_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint32_t, std::uint32_t, 4, _mm_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int32_t , std::uint32_t, 4, _mm_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, float        , std::uint32_t, 4, _mm_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint64_t, std::int32_t , 2, _mm_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int64_t , std::int32_t , 2, _mm_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, double       , std::int32_t , 2, _mm_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::uint32_t, std::int32_t , 4, _mm_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, std::int32_t , std::int32_t , 4, _mm_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx2, float        , std::int32_t , 4, _mm_i32gather_ps   ( data, ind.data.reg, 4 ) );

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX2( NAME, CMP ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint64_t, std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int64_t , std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, double       , std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint32_t, std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int32_t , std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, float        , std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint64_t, std::int64_t , 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int64_t , std::int64_t , 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, double       , std::int64_t , 4, NAME, (__m256i)_mm256_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint32_t, std::int32_t , 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int32_t , std::int32_t , 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, float        , std::int32_t , 8, NAME, (__m256i)_mm256_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint64_t, std::uint64_t, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int64_t , std::uint64_t, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, double       , std::uint64_t, 2, NAME, (__m128i)_mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint32_t, std::uint32_t, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int32_t , std::uint32_t, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, float        , std::uint32_t, 4, NAME, (__m128i)_mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint64_t, std::int64_t , 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int64_t , std::int64_t , 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, double       , std::int64_t , 2, NAME, (__m128i)_mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::uint32_t, std::int32_t , 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, std::int32_t , std::int32_t , 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx, float        , std::int32_t , 4, NAME, (__m128i)_mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX2( lt, _CMP_LT_OS )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX2( gt, _CMP_GT_OS )

////// cmp operations ------------------------------------------------------------------
////    std::uint64_t                    is_neg         () const { return _mm256_movemask_ps( (__m256)values ); }
////    // SimdVec                       permute        ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }
////    std::uint64_t                    nz             () const { return _mm256_movemask_ps( (__m256)_mm256_xor_si256( _mm256_set1_epi8( -1 ), _mm256_cmpeq_epi32( values, _mm256_setzero_si256() ) ) ); }

}
#endif // __AVX2__

} // namespace parex
