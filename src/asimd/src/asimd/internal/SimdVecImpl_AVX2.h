#pragma once

#include "SimdVecImpl_Generic.h"
#include <x86intrin.h>

namespace asimd {

#ifdef __AVX2__
namespace SimdVecInternal {

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint64_t, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int64_t , 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint32_t, 8, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int32_t , 8, anb, _mm256_and_si256 );


//// arithmetic operations that work only on int types -------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint64_t, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int64_t , 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint32_t, 8, sll, _mm256_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int32_t , 8, sll, _mm256_sllv_epi32 );

// => SSE2 size
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint64_t, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int64_t , 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::uint32_t, 4, sll, _mm_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, std::int32_t , 4, sll, _mm_sllv_epi32 );

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint64_t, std::uint32_t, 4, _mm256_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int64_t , std::uint32_t, 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, double       , std::uint32_t, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint32_t, std::uint32_t, 8, _mm256_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int32_t , std::uint32_t, 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, float        , std::uint32_t, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint64_t, std::int32_t , 4, _mm256_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int64_t , std::int32_t , 4, _mm256_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, double       , std::int32_t , 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint32_t, std::int32_t , 8, _mm256_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int32_t , std::int32_t , 8, _mm256_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, float        , std::int32_t , 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint64_t, std::uint32_t, 2, _mm_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int64_t , std::uint32_t, 2, _mm_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, double       , std::uint32_t, 2, _mm_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint32_t, std::uint32_t, 4, _mm_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int32_t , std::uint32_t, 4, _mm_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, float        , std::uint32_t, 4, _mm_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint64_t, std::int32_t , 2, _mm_i32gather_epi64( (const std::int64_t *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int64_t , std::int32_t , 2, _mm_i32gather_epi64( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, double       , std::int32_t , 2, _mm_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::uint32_t, std::int32_t , 4, _mm_i32gather_epi32( (const std::int32_t *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, std::int32_t , std::int32_t , 4, _mm_i32gather_epi32( data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, float        , std::int32_t , 4, _mm_i32gather_ps   ( data, ind.data.reg, 4 ) );

}
#endif // __AVX2__

} // namespace asimd
