#ifndef SDOT_SIMD_VEC_H
#define SDOT_SIMD_VEC_H

#include "SimdVecInternal_AVX512.h"
#include "SimdVecInternal_AVX2.h"
#include "SimdVecInternal_SSE2.h"
#include "../support/S.h"
#include "../support/N.h"
#include "SimdSize.h"

namespace parex {

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,Arch::Native>::value,class Arch=Arch::Native>
struct SimdVec {
    using                            Impl         = SimdVecInternal::Impl<T_,size_,Arch>;
    enum {                           size         = size_ };
    using                            T            = T_;

    /**/                             SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { init( impl, a, b, c, d, e, f, g, h ); }
    /**/                             SimdVec      ( T a, T b, T c, T d ) { init( impl, a, b, c, d ); }
    /**/                             SimdVec      ( T a, T b ) { init( impl, a, b ); }
    /**/                             SimdVec      ( T a ) { init( impl, a ); }
    /**/                             SimdVec      ( Impl impl ) : impl( impl ) {}
    /**/                             SimdVec      () {}

    static SimdVec                   iota         ( T beg = 0, T mul = 1 ) { return SimdVecInternal::iota( beg, mul, S<Impl>() ); }
    static SimdVec                   iota         ( T beg = 0 ) { return SimdVecInternal::iota( beg, S<Impl>() ); }

    template<class G> static SimdVec load_aligned ( const G *data ) { return SimdVecInternal::load_aligned( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data ) { return SimdVecInternal::load( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data, N<0> /*aligned*/ ) { return SimdVecInternal::load( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data, N<1> /*aligned*/ ) { return SimdVecInternal::load_aligned( data, S<Impl>() ); }

    static void                      store_aligned( T *data, const SimdVec &vec ) { SimdVecInternal::store_aligned( data, vec.impl ); }
    void                             store_aligned( T *data ) const { store_aligned( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec ) { SimdVecInternal::store( data, vec.impl ); }
    void                             store        ( T *data ) const { store( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec, N<0> /*aligned*/ ) { SimdVecInternal::store( data, vec.impl ); }
    void                             store        ( T *data, N<0> /*aligned*/ ) const { store( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec, N<1> /*aligned*/ ) { SimdVecInternal::store_aligned( data, vec.impl ); }
    void                             store        ( T *data, N<1> /*aligned*/ ) const { store_aligned( data, *this ); }

    template                         <class G,class V>
    static SimdVec                   gather       ( const G *data, const V &ind ) { return SimdVecInternal::gather( data, ind.impl, S<Impl>() ); }

    template                         <class G,class V>
    static void                      scatter      ( G *ptr, const V &ind, const SimdVec &vec ) { SimdVecInternal::scatter( ptr, ind.impl, vec.impl ); }

    const T&                         operator[]   ( int i ) const { return SimdVecInternal::at( impl, i ); }
    T&                               operator[]   ( int i ) { return SimdVecInternal::at( impl, i ); }
    const T*                         begin        () const { return &operator[]( 0 ); }
    const T*                         end          () const { return begin() + size; }

    SimdVec                          operator<<   ( const SimdVec &that ) const { return SimdVecInternal::sll( impl, that.impl ); }
    SimdVec                          operator&    ( const SimdVec &that ) const { return SimdVecInternal::anb( impl, that.impl ); }
    SimdVec                          operator+    ( const SimdVec &that ) const { return SimdVecInternal::add( impl, that.impl ); }
    SimdVec                          operator-    ( const SimdVec &that ) const { return SimdVecInternal::sub( impl, that.impl ); }
    SimdVec                          operator*    ( const SimdVec &that ) const { return SimdVecInternal::mul( impl, that.impl ); }
    SimdVec                          operator/    ( const SimdVec &that ) const { return SimdVecInternal::div( impl, that.impl ); }

    auto                             operator>    ( const SimdVec &that ) const { return SimdVecInternal::gt ( impl, that.impl );  }
    // std::uint64_t                 is_neg       () const { return value < 0; }
    // std::uint64_t                 nz           () const { return value != 0; }

    T                                sum          () const { return SimdVecInternal::horizontal_sum( impl ); }

    Impl                             impl;

};

// to convert result of things that produce boolean results (operator>, ...)
template<class SV,class OP>
SV as_SimdVec( const OP &op ) {
    return op.as_SimdVec( S<typename SV::Impl>() );
}

template<class T,int size,class Arch>
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecInternal::min( a.impl, b.impl ); }

template<class T,int size,class Arch>
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecInternal::max( a.impl, b.impl ); }


} // namespace parex

#endif // SDOT_SIMD_VEC_H
