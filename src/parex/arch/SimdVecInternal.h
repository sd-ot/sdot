#pragma once

// #include "../support/bitset.h"
#include "../support/TODO.h"
#include "../support/S.h"
#include <ostream>
#include "Arch.h"

namespace parex {
namespace SimdVecInternal {

// Impl ---------------------------------------------------------
template<class T,int size,class Arch,class Enable=void>
struct Impl {
    union {
        Impl<T,size/2,Arch> split[ 2 ];
        T values[ size ];
    } data;
};

template<class T,class Arch,class Enable>
struct Impl<T,1,Arch,Enable> {
    union {
        T values[ 1 ];
    } data;
};

/// Helper to make Impl with a register
#define SIMD_VEC_IMPL_REG( COND, T, SIZE, TREG ) \
    template<class Arch> \
    struct Impl<T,SIZE,Arch,typename std::enable_if<COND>::type> { \
        union { \
            Impl<T,(SIZE)/2,Arch> split[ 2 ]; \
            T values[ SIZE ]; \
            TREG reg; \
        } data; \
    }

// BitSet --------------------------------------------------------
//template<int size_,class Arch>
//struct BitSet {
//    enum {            size           = size_ };

//    /**/              BitSet         ( const BitSet<size/2,Arch> &a, const BitSet<size/2,Arch> &b ) : data( cat( a.data, b.data ) ) {}
//    template<class I> BitSet         ( I v ) : data( v ) {}

//    void              write_to_stream( std::ostream &os ) const { for( int i = 0; i < size; ++i ) os << data[ i ]; }
//    bool              operator[]     ( int i ) const { return data[ i ]; }
//    auto              operator[]     ( int i ) { return data[ i ]; }

//    std::bitset<size> data;
//};


// at ------------------------------------------------------------------------
template<class T,int size,class Arch>
const T &at( const Impl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

template<class T,int size,class Arch>
T &at( Impl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

// init ----------------------------------------------------------------------
template<class T,int size,class Arch,class G>
void init( Impl<T,size,Arch> &vec, G a, G b, G c, G d, G e, G f, G g, G h ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
    vec.data.values[ 5 ] = f;
    vec.data.values[ 6 ] = g;
    vec.data.values[ 7 ] = h;
}

template<class T,int size,class Arch,class G>
void init( Impl<T,size,Arch> &vec, G a, G b, G c, G d ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
}

template<class T,int size,class Arch,class G>
void init( Impl<T,size,Arch> &vec, G a, G b ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
}

template<class T,int size,class Arch,class G>
void init( Impl<T,size,Arch> &vec, G a ) {
    for( int i = 0; i < size; ++i )
        vec.data.values[ i ] = a;
}

template<class T,int size,class Arch>
void init( Impl<T,size,Arch> &vec, Impl<T,size/2,Arch> a, Impl<T,size/2,Arch> b ) {
    vec.data.split[ 0 ] = a;
    vec.data.split[ 1 ] = b;
}

#define SIMD_VEC_IMPL_REG_INIT_1( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type init( Impl<T,SIZE,Arch> &vec, T a ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_2( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type init( Impl<T,SIZE,Arch> &vec, T a, T b ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_4( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type init( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_8( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type init( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h ) { \
        vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_16( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type init( Impl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p ) { \
        vec.data.reg = FUNC; \
    }

// load_aligned( -----------------------------------------------------------------------
template<class G,class T,int size,class Arch>
Impl<T,size,Arch> load_aligned( const G *data, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split[ 0 ] = load_aligned( data + 0 * size / 2, S<Impl<T,size/2,Arch>>() );
    res.data.split[ 1 ] = load_aligned( data + 1 * size / 2, S<Impl<T,size/2,Arch>>() );
    return res;
}

template<class G,class T,class Arch>
Impl<T,1,Arch> load_aligned( const G *data, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND,Impl<T,SIZE,Arch>>::type load_aligned( const T *data, S<Impl<T,SIZE,Arch>> ) { \
        Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_FOT( COND, T, G, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND,Impl<T,SIZE,Arch>>::type load_aligned( const G *data, S<Impl<T,SIZE,Arch>> ) { \
        Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// store_aligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch>
void store_aligned( G *data, const Impl<T,size,Arch> &impl ) {
    store_aligned( data + 0 * size / 2, impl.data.split[ 0 ] );
    store_aligned( data + 1 * size / 2, impl.data.split[ 1 ] );
}

template<class G,class T,class Arch>
void store_aligned( G *data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type store_aligned( T *data, const Impl<T,SIZE,Arch> &impl ) { \
        FUNC; \
    }

// store -----------------------------------------------------------------------
template<class G,class T,int size,class Arch>
void store( G *data, const Impl<T,size,Arch> &impl ) {
    store( data + 0 * size / 2, impl.data.split[ 0 ] );
    store( data + 1 * size / 2, impl.data.split[ 1 ] );
}

template<class G,class T,class Arch>
void store( G *data, const Impl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_STORE( COND, T, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type store( T *data, const Impl<T,SIZE,Arch> &impl ) { \
        FUNC; \
    }

// arithmetic operations -------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_OP( NAME, OP ) \
    template<class T,int size,class Arch> \
    Impl<T,size,Arch> NAME( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b ) { \
        Impl<T,size,Arch> res; \
        for( int i = 0; i < 2; ++i ) \
            res.data.split[ i ] = NAME( a.data.split[ i ], b.data.split[ i ] ); \
        return res; \
    } \
    \
    template<class T,class Arch> \
    Impl<T,1,Arch> NAME( const Impl<T,1,Arch> &a, const Impl<T,1,Arch> &b ) { \
        Impl<T,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ]; \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_OP( sll, << )
    SIMD_VEC_IMPL_ARITHMETIC_OP( anb, &  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( add, +  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( sub, -  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( mul, *  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( div, /  )

#undef SIMD_VEC_IMPL_ARITHMETIC_OP

#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, SIZE, NAME, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND,Impl<T,SIZE,Arch>>::type NAME( const Impl<T,SIZE,Arch> &a, const Impl<T,SIZE,Arch> &b ) { \
        Impl<T,SIZE,Arch> res; res.data.reg = FUNC( a.data.reg, b.data.reg ); return res; \
    }

// cmp operations ------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    template<class T,int size,class Arch,class I> \
    Impl<I,size,Arch> NAME##_SimdVec( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b, S<Impl<I,size,Arch>> ) { \
        Impl<I,size,Arch> res; \
        res.data.split[ 0 ] = NAME##_SimdVec( a.data.split[ 0 ], b.data.split[ 0 ], S<Impl<I,size/2,Arch>>() ); \
        res.data.split[ 1 ] = NAME##_SimdVec( a.data.split[ 1 ], b.data.split[ 1 ], S<Impl<I,size/2,Arch>>() ); \
        return res; \
    } \
    template<class T,class Arch,class I> \
    Impl<I,1,Arch> NAME##_SimdVec( const Impl<T,1,Arch> &a, const Impl<T,1,Arch> &b, S<Impl<I,1,Arch>> ) { \
        Impl<I,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ] ? ~I( 0 ) : I( 0 ); \
        return res; \
    } \
    template<class T,int size,class Arch> \
    struct Op_##NAME { \
        template<class VI> \
        VI as_SimdVec( S<VI> ) const { \
            return NAME##_SimdVec( a, b, S<VI>() ); \
        } \
        \
        Impl<T,size,Arch> a, b; \
    }; \
    template<class T,int size,class Arch> \
    Op_##NAME<T,size,Arch> NAME( const Impl<T,size,Arch> &a, const Impl<T,size,Arch> &b ) { \
        return { a, b }; \
    }

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, I, SIZE, NAME, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND,Impl<I,SIZE,Arch>>::type NAME##_SimdVec( const Impl<T,SIZE,Arch> &a, const Impl<T,SIZE,Arch> &b, S<Impl<I,SIZE,Arch>> ) { \
        Impl<I,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// iota -----------------------------------------------------------------------------
template<class T,int size,class Arch>
Impl<T,size,Arch> iota( T beg, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split[ 0 ] = iota( beg + 0 * size / 2, S<Impl<T,size/2,Arch>>() );
    res.data.split[ 1 ] = iota( beg + 1 * size / 2, S<Impl<T,size/2,Arch>>() );
    return res;
}

template<class T,class Arch>
Impl<T,1,Arch> iota( T beg, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// sum -----------------------------------------------------------------------------
template<class T,int size,class Arch>
T sum( const Impl<T,size,Arch> &impl ) {
    return sum( impl.data.split[ 0 ] ) + sum( impl.data.split[ 1 ] );
}

template<class T,class Arch>
T sum( const Impl<T,1,Arch> &impl ) {
    return impl.data.values[ 0 ];
}

// scatter/gather -----------------------------------------------------------------------
template<class G,class V,class T,int size,class Arch>
void scatter( G *ptr, const V &ind, const Impl<T,size,Arch> &vec ) {
    scatter( ptr, ind.data.split[ 0 ], vec.data.split[ 0 ] );
    scatter( ptr, ind.data.split[ 1 ], vec.data.split[ 1 ] );
}

template<class G,class V,class T,class Arch>
void scatter( G *ptr, const V &ind, const Impl<T,1,Arch> &vec ) {
    ptr[ ind.data.values[ 0 ] ] = vec.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_SCATTER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND>::type scatter( T *data, const Impl<I,SIZE,Arch> &ind, const Impl<T,SIZE,Arch> &vec ) { \
        ; FUNC; \
    }


template<class G,class V,class T,int size,class Arch>
Impl<T,size,Arch> gather( const G *data, const V &ind, S<Impl<T,size,Arch>> ) {
    Impl<T,size,Arch> res;
    res.data.split[ 0 ] = gather( data, ind.data.split[ 0 ], S<Impl<T,size/2,Arch>>() );
    res.data.split[ 1 ] = gather( data, ind.data.split[ 1 ], S<Impl<T,size/2,Arch>>() );
    return res;
}

template<class G,class V,class T,class Arch>
Impl<T,1,Arch> gather( const G *data, const V &ind, S<Impl<T,1,Arch>> ) {
    Impl<T,1,Arch> res;
    res.data.values[ 0 ] = data[ ind.data.values[ 0 ] ];
    return res;
}

#define SIMD_VEC_IMPL_REG_GATHER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> \
    typename std::enable_if<COND,Impl<T,SIZE,Arch>>::type gather( const T *data, const Impl<I,SIZE,Arch> &ind, S<Impl<T,SIZE,Arch>> ) { \
        Impl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

} // namespace SimdVecInternal
} // namespace parex

