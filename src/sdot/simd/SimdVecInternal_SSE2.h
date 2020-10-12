#pragma once

#include "SimdVecInternal.h"
#include <x86intrin.h>
#include "SimdSize.h"

namespace sdot {

DECL_SIMD_SIZE( std::uint64_t, Arch::SSE2, 2 );
DECL_SIMD_SIZE( std::int64_t , Arch::SSE2, 2 );
DECL_SIMD_SIZE( double       , Arch::SSE2, 2 );
DECL_SIMD_SIZE( std::uint32_t, Arch::SSE2, 4 );
DECL_SIMD_SIZE( std::int32_t , Arch::SSE2, 4 );
DECL_SIMD_SIZE( float        , Arch::SSE2, 4 );

#ifdef __SSE2__
namespace SimdVecInternal {

// struct Impl<...>
SIMD_VEC_IMPL_REG( Arch::sse2, std::uint64_t, 2, __m128i );
SIMD_VEC_IMPL_REG( Arch::sse2, std::int64_t , 2, __m128i );
SIMD_VEC_IMPL_REG( Arch::sse2, double       , 2, __m128d );
SIMD_VEC_IMPL_REG( Arch::sse2, std::uint32_t, 4, __m128i );
SIMD_VEC_IMPL_REG( Arch::sse2, std::int32_t , 4, __m128i );
SIMD_VEC_IMPL_REG( Arch::sse2, float        , 4, __m128  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, std::uint64_t, 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, std::int64_t , 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, double       , 2, _mm_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, std::uint32_t, 4, _mm_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, std::int32_t , 4, _mm_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::sse2, float        , 4, _mm_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_2( Arch::sse2, std::uint64_t, 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( Arch::sse2, std::int64_t , 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( Arch::sse2, double       , 2, _mm_set_pd( b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( Arch::sse2, std::uint32_t, 4, _mm_set_epi32( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( Arch::sse2, std::int32_t , 4, _mm_set_epi32( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( Arch::sse2, float        , 4, _mm_set_ps( d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, std::uint64_t, 2, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, std::int64_t , 2, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, double       , 2, _mm_load_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, std::uint32_t, 4, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, std::int32_t , 4, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::sse2, float        , 4, _mm_load_ps( data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, std::uint64_t, 2, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, std::int64_t , 2, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, double       , 2, _mm_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, std::uint32_t, 4, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, std::int32_t , 4, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::sse2, float        , 4, _mm_store_ps   (            data, impl.data.reg ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, std::uint64_t, 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, std::int64_t , 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, double       , 2, _mm_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, std::uint32_t, 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, std::int32_t , 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::sse2, float        , 4, _mm_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::uint64_t, 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::int64_t , 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, double       , 2, NAME, _mm_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::uint32_t, 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::int32_t , 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, float        , 4, NAME, _mm_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, double, 2, NAME, _mm_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, float , 4, NAME, _mm_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::uint64_t, 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::int64_t , 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, double       , 2, anb, _mm_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::uint32_t, 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, std::int32_t , 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::sse2, float        , 4, anb, _mm_and_ps    );

////// cmp operations ------------------------------------------------------------------
// std::uint64_t                    is_neg         () const { return _mm256_movemask_ps( (__m256)values ); }
// // SimdVec                       permute        ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }
// std::uint64_t                    nz             () const { return _mm256_movemask_ps( (__m256)_mm256_xor_si256( _mm256_set1_epi8( -1 ), _mm256_cmpeq_epi32( values, _mm256_setzero_si256() ) ) ); }

}

#endif // __SSE2__

}
