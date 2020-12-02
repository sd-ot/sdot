#pragma once

#include "internal/SimdVecImpl_Generic.h"
#include "internal/SimdVecImpl_AVX512.h"
#include "internal/SimdVecImpl_AVX2.h"
#include "internal/SimdVecImpl_AVX.h"
#include "internal/SimdVecImpl_SSE2.h"
#include "internal/S.h"
#include "internal/N.h"
#include "SimdSize.h"

namespace asimd {

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,processing_units::Native>::value,class Arch=processing_units::Native>
struct SimdVec {
    using                                Impl         = SimdVecImpl::Impl<T_,size_,Arch>;
    using                                T            = T_;

     HaD                                 SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { init( impl, a, b, c, d, e, f, g, h ); }
     HaD                                 SimdVec      ( T a, T b, T c, T d ) { init( impl, a, b, c, d ); }
     HaD                                 SimdVec      ( T a, T b ) { init( impl, a, b ); }
     HaD                                 SimdVec      ( T a ) { init( impl, a ); }
     HaD                                 SimdVec      ( Impl impl ) : impl( impl ) {}
     HaD                                 SimdVec      () {}

    static HaD SimdVec                   iota         ( T beg, T mul ) { return SimdVecImpl::iota( beg, mul, S<Impl>() ); }
    static HaD SimdVec                   iota         ( T beg = 0 ) { return SimdVecImpl::iota( beg, S<Impl>() ); }

    static HaD constexpr int             size         () { return size_; }

    template<class G> static HaD SimdVec load_aligned ( const G *data ) { return SimdVecImpl::load_aligned( data, S<Impl>() ); }
    template<class G> static HaD SimdVec load         ( const G *data ) { return SimdVecImpl::load( data, S<Impl>() ); }
    template<class G> static HaD SimdVec load         ( const G *data, N<0> /*aligned*/ ) { return SimdVecImpl::load( data, S<Impl>() ); }
    template<class G> static HaD SimdVec load         ( const G *data, N<1> /*aligned*/ ) { return SimdVecImpl::load_aligned( data, S<Impl>() ); }

    static HaD void                      store_aligned( T *data, const SimdVec &vec ) { SimdVecImpl::store_aligned( data, vec.impl ); }
    HaD void                             store_aligned( T *data ) const { store_aligned( data, *this ); }

    static HaD void                      store        ( T *data, const SimdVec &vec ) { SimdVecImpl::store( data, vec.impl ); }
    HaD void                             store        ( T *data ) const { store( data, *this ); }

    static HaD void                      store        ( T *data, const SimdVec &vec, N<0> /*aligned*/ ) { SimdVecImpl::store( data, vec.impl ); }
    HaD void                             store        ( T *data, N<0> /*aligned*/ ) const { store( data, *this ); }

    static HaD void                      store        ( T *data, const SimdVec &vec, N<1> /*aligned*/ ) { SimdVecImpl::store_aligned( data, vec.impl ); }
    HaD void                             store        ( T *data, N<1> /*aligned*/ ) const { store_aligned( data, *this ); }

    template                             <class G,class V>
    static HaD SimdVec                   gather       ( const G *data, const V &ind ) { return SimdVecImpl::gather( data, ind.impl, S<Impl>() ); }

    template                             <class G,class V>
    static HaD void                      scatter      ( G *ptr, const V &ind, const SimdVec &vec ) { SimdVecImpl::scatter( ptr, ind.impl, vec.impl ); }

    HaD const T&                         operator[]   ( int i ) const { return SimdVecImpl::at( impl, i ); }
    HaD T&                               operator[]   ( int i ) { return SimdVecImpl::at( impl, i ); }
    HaD const T*                         begin        () const { return &operator[]( 0 ); }
    HaD const T*                         end          () const { return begin() + size; }

    HaD SimdVec                          operator<<   ( const SimdVec &that ) const { return SimdVecImpl::sll( impl, that.impl ); }
    HaD SimdVec                          operator&    ( const SimdVec &that ) const { return SimdVecImpl::anb( impl, that.impl ); }
    HaD SimdVec                          operator+    ( const SimdVec &that ) const { return SimdVecImpl::add( impl, that.impl ); }
    HaD SimdVec                          operator-    ( const SimdVec &that ) const { return SimdVecImpl::sub( impl, that.impl ); }
    HaD SimdVec                          operator*    ( const SimdVec &that ) const { return SimdVecImpl::mul( impl, that.impl ); }
    HaD SimdVec                          operator/    ( const SimdVec &that ) const { return SimdVecImpl::div( impl, that.impl ); }

    HaD auto                             operator>    ( const SimdVec &that ) const { return SimdVecImpl::gt ( impl, that.impl );  }

    HaD T                                sum          () const { return SimdVecImpl::horizontal_sum( impl ); }

    Impl                                 impl;
};

// to convert result of things that produce boolean results (operator>, ...)
template<class SV,class OP> HaD
SV as_SimdVec( const OP &op ) {
    return op.as_SimdVec( S<typename SV::Impl>() );
}

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::min( a.impl, b.impl ); }

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::max( a.impl, b.impl ); }


} // namespace asimd
