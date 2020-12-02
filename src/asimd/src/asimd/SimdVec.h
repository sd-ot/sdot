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
template<class T_,int size_=SimdSize<T_,InstructionSet::Native>::value,class Arch=InstructionSet::Native>
struct SimdVec {
    using                            Impl         = SimdVecImpl::Impl<T_,size_,Arch>;
    using                            T            = T_;

    /**/                             SimdVec      ( T a, T b, T c, T d, T e, T f, T g, T h ) { init( impl, a, b, c, d, e, f, g, h ); }
    /**/                             SimdVec      ( T a, T b, T c, T d ) { init( impl, a, b, c, d ); }
    /**/                             SimdVec      ( T a, T b ) { init( impl, a, b ); }
    /**/                             SimdVec      ( T a ) { init( impl, a ); }
    /**/                             SimdVec      ( Impl impl ) : impl( impl ) {}
    /**/                             SimdVec      () {}

    static SimdVec                   iota         ( T beg, T mul ) { return SimdVecImpl::iota( beg, mul, S<Impl>() ); }
    static SimdVec                   iota         ( T beg = 0 ) { return SimdVecImpl::iota( beg, S<Impl>() ); }

    static constexpr int             size         () { return size_; }

    template<class G> static SimdVec load_aligned ( const G *data ) { return SimdVecImpl::load_aligned( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data ) { return SimdVecImpl::load( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data, N<0> /*aligned*/ ) { return SimdVecImpl::load( data, S<Impl>() ); }
    template<class G> static SimdVec load         ( const G *data, N<1> /*aligned*/ ) { return SimdVecImpl::load_aligned( data, S<Impl>() ); }

    static void                      store_aligned( T *data, const SimdVec &vec ) { SimdVecImpl::store_aligned( data, vec.impl ); }
    void                             store_aligned( T *data ) const { store_aligned( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec ) { SimdVecImpl::store( data, vec.impl ); }
    void                             store        ( T *data ) const { store( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec, N<0> /*aligned*/ ) { SimdVecImpl::store( data, vec.impl ); }
    void                             store        ( T *data, N<0> /*aligned*/ ) const { store( data, *this ); }

    static void                      store        ( T *data, const SimdVec &vec, N<1> /*aligned*/ ) { SimdVecImpl::store_aligned( data, vec.impl ); }
    void                             store        ( T *data, N<1> /*aligned*/ ) const { store_aligned( data, *this ); }

    template                         <class G,class V>
    static SimdVec                   gather       ( const G *data, const V &ind ) { return SimdVecImpl::gather( data, ind.impl, S<Impl>() ); }

    template                         <class G,class V>
    static void                      scatter      ( G *ptr, const V &ind, const SimdVec &vec ) { SimdVecImpl::scatter( ptr, ind.impl, vec.impl ); }

    const T&                         operator[]   ( int i ) const { return SimdVecImpl::at( impl, i ); }
    T&                               operator[]   ( int i ) { return SimdVecImpl::at( impl, i ); }
    const T*                         begin        () const { return &operator[]( 0 ); }
    const T*                         end          () const { return begin() + size; }

    SimdVec                          operator<<   ( const SimdVec &that ) const { return SimdVecImpl::sll( impl, that.impl ); }
    SimdVec                          operator&    ( const SimdVec &that ) const { return SimdVecImpl::anb( impl, that.impl ); }
    SimdVec                          operator+    ( const SimdVec &that ) const { return SimdVecImpl::add( impl, that.impl ); }
    SimdVec                          operator-    ( const SimdVec &that ) const { return SimdVecImpl::sub( impl, that.impl ); }
    SimdVec                          operator*    ( const SimdVec &that ) const { return SimdVecImpl::mul( impl, that.impl ); }
    SimdVec                          operator/    ( const SimdVec &that ) const { return SimdVecImpl::div( impl, that.impl ); }

    auto                             operator>    ( const SimdVec &that ) const { return SimdVecImpl::gt ( impl, that.impl );  }

    T                                sum          () const { return SimdVecImpl::horizontal_sum( impl ); }

    Impl                             impl;

};

// to convert result of things that produce boolean results (operator>, ...)
template<class SV,class OP>
SV as_SimdVec( const OP &op ) {
    return op.as_SimdVec( S<typename SV::Impl>() );
}

template<class T,int size,class Arch>
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::min( a.impl, b.impl ); }

template<class T,int size,class Arch>
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return SimdVecImpl::max( a.impl, b.impl ); }


} // namespace asimd
