#pragma once

#include "../SimdRange.h"
#include "../SimdVec.h"
#include "../Math.h"

namespace asimd {

template<class A,class T,class G,class I>
typename std::enable_if<A::cpu>::type assign_scalar( const A &/*allocator*/, T *dst, const G &val, I len ) {
    SimdRange<SimdSize<T>::value>::for_each( len, [&]( I index, auto s ) {
        SimdVec<T,s.value>::store( dst + index, val );
    } );
}

#ifdef __CUDACC__
template<class T,class G,class I> __global__
void assign_scalar_gpu( T *dst, G val, I len ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < len )
        dst[ i ] = val;
}

template<class A,class T,class G,class I>
typename std::enable_if<A::gpu>::type assign_scalar( const A &/*allocator*/, T *dst, const G &val, I len ) {
    const int nb_threads = 1024;
    assign_scalar_gpu<<<div_up(len,nb_threads),nb_threads>>>( dst, val, len );
}
#endif // __CUDACC__

} // namespace asimd
