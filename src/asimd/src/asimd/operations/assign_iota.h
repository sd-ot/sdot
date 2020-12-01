#pragma once

#include "../SimdRange.h"
#include "../SimdVec.h"
#include "../Math.h"

namespace asimd {

template<class A,class T,class U,class V>
typename std::enable_if<A::cpu>::type assign_iota( const A &/*allocator*/, T *data, U beg, V len ) {
    SimdRange<SimdSize<T>::value>::template for_each_with_iota<T>( 0, len, [&]( T index, auto iota, auto s ) {
        SimdVec<T,s.value>::store( data + index, iota ); // _aligned
    }, beg );
}

#ifdef __CUDACC__
template<class T,class G,class I> __global__
void assign_iota_gpu( T *dst, G beg, I len ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < len )
        dst[ i ] = beg + i;
}

template<class A,class T,class G,class I>
typename std::enable_if<A::gpu>::type assign_iota( const A &/*allocator*/, T *dst, G beg, I len ) {
    const int nb_threads = 1024;
    assign_iota_gpu<<<div_up(len,nb_threads),nb_threads>>>( dst, beg, len );
}
#endif // __CUDACC__


} // namespace asimd
