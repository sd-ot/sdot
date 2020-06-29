#pragma once

#include "SimdVec.h"

/// generic, with only 1 value
template<class T,class Arch>
struct SimdVec<T,1,Arch> {
    /**/                             SimdVec        ( T value ) : value( value ) {}
    /**/                             SimdVec        () {}

    template<class I> static void    store_with_mask( T *data, const SimdVec &vec, I mask ) { if ( mask & 1 ) *( data++ ) = vec[ 0 ]; }
    static void                      store_aligned  ( T *data, const SimdVec &vec ) { *data = vec.value; }
    template<class G> static SimdVec load_aligned   ( const G *values ) { return *values; }
    template<class M> static SimdVec iota_mask      ( M mask, T &n ) { bool res = mask & 1; n += res; return res; }
    template<class V> static void    scatter        ( T *ptr, const V &ind, const SimdVec &vec ) { ptr[ ind[ 0 ] ] = vec[ 0 ]; }
    template<class V> static SimdVec gather         ( const T *values, const V &ind ) { return values[ ind[ 0 ] ]; }
    static SimdVec                   iota           ( T beg ) { return beg; }

    SimdVec                          operator<<     ( const SimdVec &that ) const { return value << that.value; }
    SimdVec                          operator&      ( const SimdVec &that ) const { return value &  that.value; }
    SimdVec                          operator+      ( const SimdVec &that ) const { return value +  that.value; }
    SimdVec                          operator-      ( const SimdVec &that ) const { return value -  that.value; }
    SimdVec                          operator*      ( const SimdVec &that ) const { return value *  that.value; }
    SimdVec                          operator/      ( const SimdVec &that ) const { return value /  that.value; }

    std::uint64_t                    operator>      ( const SimdVec &that ) const { return value >  that.value;  }
    std::uint64_t                    is_neg         () const { return value < 0; }
    std::uint64_t                    nz             () const { return value != 0; }

    T                                operator[]     ( int /*n*/ ) const { return value; }
    T&                               operator[]     ( int /*n*/ ) { return value; }
    const T*                         begin          () const { return &value; }
    const T*                         end            () const { return &value + 1; }

    T                                value;
};
