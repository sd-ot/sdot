#pragma once

#pragma once

#include "SimdVecImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {

#ifdef __AVX__
namespace SimdVecInternal {

// struct Impl<...>
SIMD_VEC_IMPL_REG( AVX, std::uint64_t, 4, __m256i );
SIMD_VEC_IMPL_REG( AVX, std::int64_t , 4, __m256i );
SIMD_VEC_IMPL_REG( AVX, double       , 4, __m256d );
SIMD_VEC_IMPL_REG( AVX, std::uint32_t, 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, std::int32_t , 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, float        , 8, __m256  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( AVX, std::uint64_t, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, std::int64_t , 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, double       , 4, _mm256_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, std::uint32_t, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, std::int32_t , 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, float        , 8, _mm256_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_4( AVX, std::uint64_t, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, std::int64_t , 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, double       , 4, _mm256_set_pd( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, std::uint32_t, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, std::int32_t , 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, float        , 8, _mm256_set_ps( h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, std::uint64_t, 4, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, std::int64_t , 4, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, double       , 4, _mm256_load_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, std::uint32_t, 8, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, std::int32_t , 8, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, float        , 8, _mm256_load_ps( data ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD( AVX, std::uint64_t, 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( AVX, std::int64_t , 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( AVX, double       , 4, _mm256_loadu_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD( AVX, std::uint32_t, 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( AVX, std::int32_t , 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD( AVX, float        , 8, _mm256_loadu_ps( data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, std::uint64_t, 4, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, std::int64_t , 4, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, double       , 4, _mm256_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, std::uint32_t, 8, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, std::int32_t , 8, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, float        , 8, _mm256_store_ps   (            data, impl.data.reg ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE( AVX, std::uint64_t, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( AVX, std::int64_t , 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( AVX, double       , 4, _mm256_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( AVX, std::uint32_t, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( AVX, std::int32_t , 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( AVX, float        , 8, _mm256_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, std::uint64_t, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, std::int64_t , 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, double       , 4, NAME, _mm256_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, std::uint32_t, 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, std::int32_t , 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, float        , 8, NAME, _mm256_##NAME##_ps    );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( add );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( sub );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( min );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, double, 4, NAME, _mm256_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, float , 8, NAME, _mm256_##NAME##_ps );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( mul );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , double       , 4, anb, _mm256_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , float        , 8, anb, _mm256_and_ps    );

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( NAME, CMP ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint64_t, std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int64_t , std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, double       , std::uint64_t, 4, NAME, (__m256i)_mm256_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint32_t, std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int32_t , std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, float        , std::uint32_t, 8, NAME, (__m256i)_mm256_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint64_t, std::int64_t , 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int64_t , std::int64_t , 4, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, double       , std::int64_t , 4, NAME, (__m256i)_mm256_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint32_t, std::int32_t , 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int32_t , std::int32_t , 8, NAME, (__m256i)_mm256_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, float        , std::int32_t , 8, NAME, (__m256i)_mm256_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint64_t, std::uint64_t, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int64_t , std::uint64_t, 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, double       , std::uint64_t, 2, NAME, (__m128i)_mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint32_t, std::uint32_t, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int32_t , std::uint32_t, 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, float        , std::uint32_t, 4, NAME, (__m128i)_mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint64_t, std::int64_t , 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int64_t , std::int64_t , 2, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, double       , std::int64_t , 2, NAME, (__m128i)_mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::uint32_t, std::int32_t , 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, std::int32_t , std::int32_t , 4, NAME, (__m128i)_mm_cmp_epi64( a.data.reg, b.data.reg, CMP ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, float        , std::int32_t , 4, NAME, (__m128i)_mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( lt, _CMP_LT_OS )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( gt, _CMP_GT_OS )

}
#endif // __AVX__

} // namespace asimd
