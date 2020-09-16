#pragma once

#include "SimdVecInternal.h"
#include "../MachineArch.h"
#include <x86intrin.h>
#include "SimdSize.h"

namespace sdot {

DECL_SIMD_SIZE( std::uint64_t, MachineArch::AVX512,  8 );
DECL_SIMD_SIZE( std::int64_t , MachineArch::AVX512,  8 );
DECL_SIMD_SIZE( double       , MachineArch::AVX512,  8 );
DECL_SIMD_SIZE( std::uint32_t, MachineArch::AVX512, 16 );
DECL_SIMD_SIZE( std::int32_t , MachineArch::AVX512, 16 );
DECL_SIMD_SIZE( float        , MachineArch::AVX512, 16 );

namespace SimdVecInternal {

#ifdef __AVX512F__

// struct Impl<...>
SIMD_VEC_IMPL_REG( Arch::avx512, std::uint64_t,  8, __m512i );
SIMD_VEC_IMPL_REG( Arch::avx512, std::int64_t ,  8, __m512i );
SIMD_VEC_IMPL_REG( Arch::avx512, double       ,  8, __m512d );
SIMD_VEC_IMPL_REG( Arch::avx512, std::uint32_t, 16, __m512i );
SIMD_VEC_IMPL_REG( Arch::avx512, std::int32_t , 16, __m512i );
SIMD_VEC_IMPL_REG( Arch::avx512, float        , 16, __m512  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, std::uint64_t,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, std::int64_t ,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, double       ,  8, _mm512_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, std::uint32_t, 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, std::int32_t , 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( Arch::avx512, float        , 16, _mm512_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_8 ( Arch::avx512, std::uint64_t,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( Arch::avx512, std::int64_t ,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( Arch::avx512, double       ,  8, _mm512_set_pd   ( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( Arch::avx512, std::uint32_t, 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( Arch::avx512, std::int32_t , 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( Arch::avx512, float        , 16, _mm512_set_ps   ( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, std::uint64_t,  8, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, std::int64_t ,  8, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, double       ,  8, _mm512_load_pd( data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, std::uint32_t, 16, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, std::int32_t , 16, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( Arch::avx512, float        , 16, _mm512_load_ps( data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, std::uint64_t,  8, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, std::int64_t ,  8, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, double       ,  8, _mm512_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, std::uint32_t, 16, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, std::int32_t , 16, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( Arch::avx512, float        , 16, _mm512_store_ps   (            data, impl.data.reg ) );

// store -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, std::uint64_t,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, std::int64_t ,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, double       ,  8, _mm512_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, std::uint32_t, 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, std::int32_t , 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE( Arch::avx512, float        , 16, _mm512_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint64_t,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int64_t ,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, double       ,  8, NAME, _mm512_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint32_t, 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int32_t , 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, float        , 16, NAME, _mm512_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, double,  8, NAME, _mm512_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, float , 16, NAME, _mm512_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint64_t,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int64_t ,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, double       ,  8, anb, _mm512_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint32_t, 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int32_t , 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, float        , 16, anb, _mm512_and_ps    );

//// arithmetic operations that work only on int types ------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint64_t,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int64_t ,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::uint32_t, 16, sll, _mm512_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( Arch::avx512, std::int32_t , 16, sll, _mm512_sllv_epi32 );

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::uint64_t, std::uint32_t,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::int64_t , std::uint32_t,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, double       , std::uint32_t,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::uint32_t, std::uint32_t, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::int32_t , std::uint32_t, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, float        , std::uint32_t, 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::uint64_t, std::int32_t ,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::int64_t , std::int32_t ,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, double       , std::int32_t ,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::uint32_t, std::int32_t , 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, std::int32_t , std::int32_t , 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( Arch::avx512, float        , std::int32_t , 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

// scatter -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::uint32_t,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::uint32_t,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::uint32_t,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::uint32_t, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::uint32_t, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::uint32_t, 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::int32_t ,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::int32_t ,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::int32_t ,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::int32_t , 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::int32_t , 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::int32_t , 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// AVX2 sizes
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::uint32_t, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::uint32_t, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::uint32_t, 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::uint32_t, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::uint32_t, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::uint32_t, 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::int32_t , 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::int32_t , 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::int32_t , 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::int32_t , 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::int32_t , 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::int32_t , 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::uint32_t, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::uint32_t, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::uint32_t, 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::uint32_t, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::uint32_t, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::uint32_t, 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint64_t, std::int32_t , 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int64_t , std::int32_t , 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, double       , std::int32_t , 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::uint32_t, std::int32_t , 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, std::int32_t , std::int32_t , 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( Arch::avx512, float        , std::int32_t , 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( NAME, FLAG_F, FLAG_I ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint64_t, std::uint32_t,  8, NAME, _mm256_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int64_t , std::uint32_t,  8, NAME, _mm256_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, double       , std::uint32_t,  8, NAME, _mm256_movm_epi32( _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint32_t, std::uint32_t, 16, NAME, _mm512_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int32_t , std::uint32_t, 16, NAME, _mm512_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, float        , std::uint32_t, 16, NAME, _mm512_movm_epi32( _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint64_t, std::int32_t ,  8, NAME, _mm256_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int64_t , std::int32_t ,  8, NAME, _mm256_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, double       , std::int32_t ,  8, NAME, _mm256_movm_epi32( _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint32_t, std::int32_t , 16, NAME, _mm512_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int32_t , std::int32_t , 16, NAME, _mm512_movm_epi32( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, float        , std::int32_t , 16, NAME, _mm512_movm_epi32( _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint64_t, std::uint64_t, 8, NAME, _mm512_movm_epi64( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int64_t , std::uint64_t, 8, NAME, _mm512_movm_epi64( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, double       , std::uint64_t, 8, NAME, _mm512_movm_epi64( _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \
    \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::uint64_t, std::int64_t,  8, NAME, _mm512_movm_epi64( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, std::int64_t , std::int64_t,  8, NAME, _mm512_movm_epi64( _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( Arch::avx512, double       , std::int64_t,  8, NAME, _mm512_movm_epi64( _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ) ); \

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( lt, _CMP_LT_OS, _MM_CMPINT_LT )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( gt, _CMP_GT_OS, _MM_CMPINT_GT )

#endif // __AVX512F__

} // namespace SimdVecInternal
} // namespace sdot
