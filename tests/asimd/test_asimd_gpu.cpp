#include <asimd/SimdVec.h>
#include <cuda_runtime.h>
#include "catch_main.h"
#include "P.h"

//// nsmake cxx_name nvcc
//// nsmake cpp_flag -x
//// nsmake cpp_flag cu

using namespace asimd;

template<class T> __global__
void test( T *dst, int len ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < len ) {
        SimdVec<T,1> v = SimdVec<T,1>::iota( i );
        SimdVec<T,1>::store_aligned( dst + i, v * 5 );
    }
}

TEST_CASE( "Range", "[asimd]" ) {
    const int nb_threads = 1024, len = 11;

    int *dst_gpu;
    cudaMalloc( &dst_gpu, sizeof( int ) * len );
    test<<<( len + nb_threads - 1 ) / nb_threads,nb_threads>>>( dst_gpu, len );

    std::vector<int> dst_cpu( len );
    cudaMemcpy( dst_cpu.data(), dst_gpu, sizeof( int ) * len, cudaMemcpyDeviceToHost );

    P( dst_cpu );
}
