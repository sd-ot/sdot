#ifndef SDOT_SIMD_VEC_H
#define SDOT_SIMD_VEC_H

#include "SimdVecInternal_AVX512.h"
#include "SimdVecInternal_AVX2.h"
#include "SimdVecInternal_SSE2.h"
#include "SimdSize.h"

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,CpuArch::Native>::value,class Arch=CpuArch::Native>
struct SimdVec {
    using          Impl         = SimdVecInternal::Impl<T_,size_,Arch>;
    enum {         size         = size_ };
    using          T            = T_;

    /**/           SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { init( impl, a, b, c, d, e, f, g, h ); }
    /**/           SimdVec      ( T a, T b, T c, T d ) { init( impl, a, b, c, d ); }
    /**/           SimdVec      ( T a, T b ) { init( impl, a, b ); }
    /**/           SimdVec      ( T a ) { init( impl, a ); }
    /**/           SimdVec      ( Impl impl ) : impl( impl ) {}
    /**/           SimdVec      () {}

    template       <class G>
    static SimdVec load_aligned ( const G *data ) { return SimdVecInternal::load_aligned( data, S<Impl>() ); }
    template       <class G,class V>
    static SimdVec gather       ( const G *data, const V &ind ) { return SimdVecInternal::gather( data, ind.impl, S<Impl>() ); }
    static SimdVec iota         ( T beg = 0 ) { return SimdVecInternal::iota( beg, S<Impl>() ); }

    static void    store_aligned( T *data, const SimdVec &vec ) { SimdVecInternal::store_aligned( data, vec.impl ); }
    void           store_aligned( T *data ) const { store_aligned( data, *this ); }

    static void    store( T *data, const SimdVec &vec ) { SimdVecInternal::store( data, vec.impl ); }
    void           store( T *data ) const { store( data, *this ); }

    template       <class G,class V>
    static void    scatter      ( G *ptr, const V &ind, const SimdVec &vec ) { SimdVecInternal::scatter( ptr, ind.impl, vec.impl ); }

    const T&       operator[]   ( int i ) const { return SimdVecInternal::at( impl, i ); }
    T&             operator[]   ( int i ) { return SimdVecInternal::at( impl, i ); }
    const T*       begin        () const { return &operator[]( 0 ); }
    const T*       end          () const { return begin() + size; }

    SimdVec        operator<<   ( const SimdVec &that ) const { return SimdVecInternal::sll( impl, that.impl ); }
    SimdVec        operator&    ( const SimdVec &that ) const { return SimdVecInternal::anb( impl, that.impl ); }
    SimdVec        operator+    ( const SimdVec &that ) const { return SimdVecInternal::add( impl, that.impl ); }
    SimdVec        operator-    ( const SimdVec &that ) const { return SimdVecInternal::sub( impl, that.impl ); }
    SimdVec        operator*    ( const SimdVec &that ) const { return SimdVecInternal::mul( impl, that.impl ); }
    SimdVec        operator/    ( const SimdVec &that ) const { return SimdVecInternal::div( impl, that.impl ); }

    auto           operator>    ( const SimdVec &that ) const { return SimdVecInternal::gt ( impl, that.impl );  }
    // std::uint64_t is_neg     () const { return value < 0; }
    // std::uint64_t nz         () const { return value != 0; }

    T              sum          () const { return SimdVecInternal::sum( impl ); }

    Impl           impl;

};

// to convert result of things that produce boolean results (operator>, ...)
template<class SV,class OP>
SV as_SimdVec( const OP &op ) {
    return op.as_SimdVec( S<typename SV::Impl>() );
}

#endif // SDOT_SIMD_VEC_H

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<std::uint32_t,4,Arch,CpuArch::SSE2> {
//    using                            T              = std::uint32_t;
//    enum {                           size           = 4 };

//    // static SimdVec                load_aligned   ( const std::uint8_t *data ) { return _mm256_cvtepi8_epi64( _mm_set1_epi32( *reinterpret_cast<const std::uint32_t *>( data ) ) ); }
//    /**/                             SimdVec        ( __m128i values ) : values( values ) {}
//    /**/                             SimdVec        ( T a, T b, T c, T d ) { values = _mm_set_epi32( d, c, b, a ); }
//    /**/                             SimdVec        ( T value ) { values = _mm_set1_epi32( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm_store_si128( (__m128i *)data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm_load_si128( reinterpret_cast<const __m128i *>( data ) ); }
//    //static SimdVec                 from_int8s     ( std::uint64_t val ) { return _mm_cvtsi64_si128( val ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm_add_epi32( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm_sub_epi32( values, that.values ); }
//    // SimdVec                       operator*      ( const SimdVec &that ) const { return _mm_mul_epi32( values, that.values; }
//    // SimdVec                       operator/      ( const SimdVec &that ) const { return _mm_div_epi32( values, that.values; }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return _mm_sllv_epi32( values, that.values ); }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return _mm_and_si128( values, that.values ); }

//    std::uint64_t                    is_neg         () const { return _mm_movemask_ps( (__m128)values ); }
//    std::uint64_t                    nz             () const { return _mm_movemask_ps( (__m128)_mm_xor_si128( _mm_set1_epi8( -1 ), _mm_cmpeq_epi32( values, _mm_setzero_si128() ) ) ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T*                         begin          () const { return reinterpret_cast<const T *>( this ); }
//    const T*                         end            () const { return begin() + size; }

//    __m128i                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<double,2,Arch,CpuArch::SSE2> {
//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm_store_pd( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm_load_pd( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm_add_pd( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm_sub_pd( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm_mul_pd( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm_div_pd( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { __m128d c = _mm_cmpgt_pd( values, that.values ); return _mm_movemask_pd( c );  }
//    std::uint64_t                    is_neg         () const { return _mm_movemask_pd( values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T*                         begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T*                         end            () const { return begin() + size; }

//    __m128d                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<float,4,Arch,CpuArch::SSE2> {
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm_store_ps( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm_load_ps( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm_add_ps( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm_sub_ps( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm_mul_ps( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm_div_ps( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { __m128 c = _mm_cmpgt_ps( values, that.values ); return _mm_movemask_ps( c );  }
//    std::uint64_t                    is_neg         () const { return _mm_movemask_ps( values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T*                         begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T*                         end            () const { return begin() + size; }

//    __m128                           values;
//};
//#endif

//#ifdef __AVX2__
//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<std::uint64_t,4,Arch,CpuArch::AVX2> {
//    enum {                           size           = 4 };
//    using                            T              = std::uint64_t;

//    /**/                             SimdVec        ( __m256i values ) : values( values ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm256_set1_epi64x( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm256_store_si256( reinterpret_cast<__m256i *>( data ), vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm256_load_si256( reinterpret_cast<const __m256i *>( data ) ); }
//    static SimdVec                   load_aligned   ( std::uint8_t *data ) { return _mm256_cvtepi8_epi64( _mm_set1_epi32( *reinterpret_cast<const std::uint32_t *>( data ) ) ); }
//    static SimdVec                   from_int8s     ( std::uint64_t val ) { return _mm256_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm256_add_epi64( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm256_sub_epi64( values, that.values ); }
//    //SimdVec                        operator*      ( const SimdVec &that ) const { return _mm256_mul_epi64( values, that.values ); }
//    //SimdVec                        operator/      ( const SimdVec &that ) const { return _mm256_add_epi64( values, that.values ); }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return _mm256_sllv_epi64( values, that.values ); }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return _mm256_and_si256( values, that.values ); }

//    std::uint64_t                    is_neg         () const { return _mm256_movemask_pd( (__m256d)values ); }
//    std::uint64_t                    nz             () const { return _mm256_movemask_pd( (__m256d)_mm256_xor_si256( _mm256_set1_epi8( -1 ), _mm256_cmpeq_epi64( values, _mm256_setzero_si256() ) ) ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m256i                          values;
//};

//template<class Arch> struct SimdVec<std::uint32_t,8,Arch,CpuArch::AVX2> {
//    enum {                           size           = 8 };
//    using                            T              = std::uint32_t;

//    /**/                             SimdVec        ( __m256i values ) : values( values ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm256_set1_epi32( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm256_store_si256( reinterpret_cast<__m256i *>( data ), vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm256_load_si256( reinterpret_cast<const __m256i *>( data ) ); }
//    // static SimdVec                load_aligned   ( std::uint8_t *data ) { return _mm256_cvtepi8_epi64( _mm_set1_epi32( *reinterpret_cast<const std::uint32_t *>( data ) ) ); }
//    // static SimdVec                from_int8s     ( std::uint64_t val ) { return _mm256_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm256_add_epi32( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm256_sub_epi32( values, that.values ); }
//    //SimdVec                        operator*      ( const SimdVec &that ) const { return _mm256_mul_epi32( values, that.values ); }
//    //SimdVec                        operator/      ( const SimdVec &that ) const { return _mm256_add_epi32( values, that.values ); }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return _mm256_sllv_epi32( values, that.values ); }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return _mm256_and_si256( values, that.values ); }

//    std::uint64_t                    is_neg         () const { return _mm256_movemask_ps( (__m256)values ); }
//    std::uint64_t                    nz             () const { return _mm256_movemask_ps( (__m256)_mm256_xor_si256( _mm256_set1_epi8( -1 ), _mm256_cmpeq_epi32( values, _mm256_setzero_si256() ) ) ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,4> idx, SimdVec b ) const { return _mm256_permutex2var_pd( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m256i                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<double,4,Arch,CpuArch::AVX2> {
//    enum {                           size           = 4 };
//    using                            T              = double;

//    /**/                             SimdVec        ( __m256d values ) : values( values ) {}
//    /**/                             SimdVec        ( double a, double b, double c, double d ) : values( _mm256_set_pd( d, c, b, a ) ) {}
//    ///**/                           SimdVec        ( SimdVec<double,2> a, SimdVec<double,2> b ) : values( _mm256_set_m128d( b.values, a.values ) ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm256_set1_pd( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm256_store_pd( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm256_load_pd( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm256_add_pd( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm256_sub_pd( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm256_mul_pd( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm256_div_pd( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { __m256d c = _mm256_cmp_pd( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_pd( c );  }
//    std::uint64_t                    is_neg         () const { return _mm256_movemask_pd( values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m256d                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<float,8,Arch,CpuArch::AVX2> {
//    enum {                           size           = 8 };
//    using                            T              = float;

//    /**/                             SimdVec        ( __m128 a, __m128 b ) { for( int i = 0; i < size / 2; ++i ) { values[ i ] = a[ i ]; values[ i + size / 2 ] = b[ i ];  } }
//    /**/                             SimdVec        ( __m256 values ) : values( values ) {}
//    /**/                             SimdVec        ( float a, float b, float c, float d, float e, float f, float g, float h ) : values( _mm256_set_ps( h, g, f, e, d, c, b, a ) ) {}
//    ///**/                           SimdVec        ( SimdVec<double,2> a, SimdVec<double,2> b ) : values( _mm256_set_m128d( b.values, a.values ) ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm256_set1_ps( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm256_store_ps( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm256_load_ps( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   gather         ( const T *values, const SimdVec<std::uint32_t,8,Arch,CpuArch::AVX2> &ind ) { return _mm256_i32gather_ps( values, ind.values, sizeof( T ) ); }
//    static SimdVec                   gather         ( const T *values, const SimdVec<std::uint64_t,8,Arch,CpuArch::AVX2> &ind ) { return { _mm256_i64gather_ps( values, ind.values[ 0 ].values, sizeof( T ) ), _mm256_i64gather_ps( values, ind.values[ 1 ].values, sizeof( T ) ) }; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm256_add_ps( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm256_sub_ps( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm256_mul_ps( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm256_div_ps( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { __m256 c = _mm256_cmp_ps( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_ps( c );  }
//    std::uint64_t                    is_neg         () const { return _mm256_movemask_ps( values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m256                           values;
//};
//#endif


//#ifdef __AVX512F__
//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<std::uint64_t,8,Arch,CpuArch::AVX512> {
//    enum {                           size           = 8 };
//    using                            T              = std::uint64_t;

//    /**/                             SimdVec        ( __m512i values ) : values( values ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm512_set1_epi64( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm512_store_epi64( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm512_load_epi64( data ); }
//    static SimdVec                   load_aligned   ( const std::uint8_t *data ) { return _mm512_cvtepi8_epi64( _mm_set1_epi64x( *reinterpret_cast<const std::uint64_t *>( data ) ) ); }
//    static SimdVec                   from_int8s     ( std::uint64_t val ) { return _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm512_add_epi64( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm512_sub_epi64( values, that.values ); }
//    //SimdVec                        operator*      ( const SimdVec &that ) const { return _mm512_mul_epi64( values, that.values ); }
//    //SimdVec                        operator/      ( const SimdVec &that ) const { return _mm512_div_epi64( values, that.values ); }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return _mm512_sllv_epi64( values, that.values ); }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return _mm512_and_epi64( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { return _mm512_cmp_epi64_mask( values, that.values, _CMP_GT_OQ );  }
//    std::uint64_t                    is_neg         () const { return _mm512_movepi64_mask( values ); }
//    std::uint64_t                    nz             () const { return _mm512_cmpneq_epi64_mask( values, _mm512_setzero_si512() ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_epi64( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( this ); }
//    const T                         *end            () const { return begin() + size; }

//    __m512i                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<std::uint32_t,16,Arch,CpuArch::AVX512> {
//    enum {                           size           = 16 };
//    using                            T              = std::uint32_t;

//    /**/                             SimdVec        ( __m512i values ) : values( values ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm512_set1_epi32( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm512_store_epi32( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm512_load_epi32( data ); }
//    static SimdVec                   load_aligned   ( const std::uint8_t *data ) { return _mm512_cvtepi8_epi64( _mm_set1_epi64x( *reinterpret_cast<const std::uint64_t *>( data ) ) ); }
//    //static SimdVec                 from_int8s     ( std::uint64_t val ) { return _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( val ) ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm512_add_epi32( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm512_sub_epi32( values, that.values ); }
//    //SimdVec                        operator*      ( const SimdVec &that ) const { return _mm512_mul_epi32( values, that.values ); }
//    //SimdVec                        operator/      ( const SimdVec &that ) const { return _mm512_div_epi32( values, that.values ); }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return _mm512_sllv_epi32( values, that.values ); }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return _mm512_and_epi32( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { return _mm512_cmp_epi32_mask( values, that.values, _CMP_GT_OQ );  }
//    std::uint64_t                    is_neg         () const { return _mm512_movepi32_mask( values ); }
//    std::uint64_t                    nz             () const { return _mm512_cmpneq_epi32_mask( values, _mm512_setzero_si512() ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_epi64( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m512i                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<double,8,Arch,CpuArch::AVX512> {
//    enum {                           size           = 8 };
//    using                            T              = double;

//    /**/                             SimdVec        ( __m512d values ) : values( values ) {}
//    /**/                             SimdVec        ( T a, T b, T c, T d, T e, T f, T g, T h ) { values = _mm512_set_pd( h, g, f, e, d, c, b, a ); }
//    /**/                             SimdVec        ( T value ) { values = _mm512_set1_pd( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm512_store_pd( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm512_load_pd( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm512_add_pd( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm512_sub_pd( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm512_mul_pd( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm512_div_pd( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { return _mm512_cmp_pd_mask( values, that.values, _CMP_GT_OQ );  }
//    std::uint64_t                    is_neg         () const { return _mm512_movepi64_mask( (__m512i)values ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_pd( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m512d                          values;
//};

//// ------------------------------------------------------------------------------------------------------------------
//template<class Arch> struct SimdVec<float,16,Arch,CpuArch::AVX512> {
//    enum {                           size           = 16 };
//    using                            T              = float;

//    /**/                             SimdVec        ( __m256 a, __m256 b ) { for( int i = 0; i < size / 2; ++i ) { values[ i ] = a[ i ]; values[ i + size / 2 ] = b[ i ];  } }
//    /**/                             SimdVec        ( __m512 values ) : values( values ) {}
//    /**/                             SimdVec        ( T value ) { values = _mm512_set1_ps( value ); }
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) *( data++ ) = vec[ i ]; }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { _mm512_store_ps( data, vec.values ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return _mm512_load_ps( data ); }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = values[ ind[ i ] ]; return res; }
//    static SimdVec                   gather         ( const T *values, const SimdVec<std::uint32_t,16,Arch,CpuArch::AVX512> &ind ) { return _mm512_i32gather_ps( ind.values, values, sizeof( T ) ); }
//    static SimdVec                   gather         ( const T *values, const SimdVec<std::uint64_t,16,Arch,CpuArch::AVX512> &ind ) { return { _mm512_i64gather_ps( ind.values[ 0 ].values, values, sizeof( T ) ), _mm512_i64gather_ps( ind.values[ 1 ].values, values, sizeof( T ) ) }; }
//    static SimdVec                   iota           ( T beg ) { SimdVec res; for( int i = 0; i < size; ++i ) res[ i ] = beg + i; return res; }

//    SimdVec                          operator+      ( const SimdVec &that ) const { return _mm512_add_ps( values, that.values ); }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return _mm512_sub_ps( values, that.values ); }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return _mm512_mul_ps( values, that.values ); }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return _mm512_div_ps( values, that.values ); }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { return _mm512_cmp_ps_mask( values, that.values, _CMP_GT_OQ );  }
//    std::uint64_t                    is_neg         () const { return _mm512_movepi32_mask( (__m512i)values ); }

//    // SimdVec                       permute        ( SimdVec<std::uint64_t,8> idx, SimdVec b ) const { return _mm512_permutex2var_ps( values, idx.values, b.values ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T                         *begin          () const { return reinterpret_cast<const T *>( &values ); }
//    const T                         *end            () const { return begin() + size; }

//    __m512                           values;
//};

//#endif

//// ------------------------------------------------------------------------------------------------------------------
///// generic, several values (split in 2 terms)
//template<class T,int size,class Arch,class UsedArch>
//struct SimdVec {
//    enum {                           mi             = size / 2 };
//    using                            N              = SimdVec<T,mi,Arch,UsedArch>;

//    /**/                             SimdVec        ( N a, N b ) : values{ a, b } {}
//    /**/                             SimdVec        ( T value ) : values{ value, value } {}
//    /**/                             SimdVec        () {}

//    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { N::store_with_mask( data, vec.values[ 0 ], mask ); N::store_with_mask( data + popcnt( mask & ( ( 1 << mi ) - 1 ), S<Arch>() ), vec.values[ 1 ], mask >> mi ); }
//    static void                      store_aligned  ( T *data, const SimdVec &vec ) { N::store_aligned( data, vec.values[ 0 ] ); N::store_aligned( data + mi, vec.values[ 1 ] ); }
//    static SimdVec                   load_aligned   ( const T *data ) { return { N::load_aligned( data ), N::load_aligned( data + mi ) }; }
//    template<class M> static SimdVec iota_mask      ( M mask, T &n ) { T o = 0; SimdVec res; for( int i = 0; i < size; ++i, mask >>= 1 ) if ( mask & 1 ) res[ o++ ] = i; n += o; for( int i = o; i < size; ++i ) res[ i ] = o; return res; }
//    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) ptr[ ind[ i ] ] = vec[ i ]; }
//    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { return { N::gather( values, ind.values[ 0 ] ), N::gather( values, ind.values[ 1 ] ) }; }
//    static SimdVec                   iota           ( T beg ) { return { N::iota( beg ), N::iota( beg + mi ) }; }

//    SimdVec                          operator<<     ( const SimdVec &that ) const { return { values[ 0 ] << that.values[ 0 ], values[ 1 ] << that.values[ 1 ] }; }
//    SimdVec                          operator&      ( const SimdVec &that ) const { return { values[ 0 ] &  that.values[ 0 ], values[ 1 ] &  that.values[ 1 ] }; }
//    SimdVec                          operator+      ( const SimdVec &that ) const { return { values[ 0 ] +  that.values[ 0 ], values[ 1 ] +  that.values[ 1 ] }; }
//    SimdVec                          operator-      ( const SimdVec &that ) const { return { values[ 0 ] -  that.values[ 0 ], values[ 1 ] -  that.values[ 1 ] }; }
//    SimdVec                          operator*      ( const SimdVec &that ) const { return { values[ 0 ] *  that.values[ 0 ], values[ 1 ] *  that.values[ 1 ] }; }
//    SimdVec                          operator/      ( const SimdVec &that ) const { return { values[ 0 ] /  that.values[ 0 ], values[ 1 ] /  that.values[ 1 ] }; }

//    std::uint64_t                    operator>      ( const SimdVec &that ) const { return ( values[ 0 ] > that.values[ 0 ] ) + ( ( values[ 1 ] > that.values[ 1 ] ) << mi ); }
//    std::uint64_t                    is_neg         () const { return values[ 0 ].is_neg() + ( values[ 1 ].is_neg() << mi ); }
//    std::uint64_t                    nz             () const { return values[ 0 ].nz() + ( values[ 1 ].nz() << mi ); }

//    T                                operator[]     ( int n ) const { return reinterpret_cast<const T *>( &values )[ n ]; }
//    T&                               operator[]     ( int n ) { return reinterpret_cast<T *>( &values )[ n ]; }
//    const T*                         begin          () const { return values[ 0 ].begin(); }
//    const T*                         end            () const { return values[ 1 ].end(); }

//    N                                values[ 2 ];
//};

