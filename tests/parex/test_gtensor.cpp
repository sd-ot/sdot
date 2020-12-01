#include <asimd/operations/assign_scalar.h>
#include <asimd/allocators/GpuAllocator.h>
#include <parex/containers/gtensor.h>
#include <parex/P.h>

// using namespace parex;

/// nsmake cxx_name nvcc

int main() {
    //    using Al = asimd::GpuAllocator<double>;
    using Al = asimd::GpuAllocator<double>;
    Al al;

    gtensor<double,3,Al> t;
    t.resize( al, 2, 3, 4 );
    P( t.shape() );

    for( std::size_t i = 0; i < t.shape( 0 ); ++i )
        for( std::size_t j = 0; j < t.shape( 1 ); ++j )
            asimd::assign_scalar( t.data( i, j, 0 ), 10 * i + j, 4 );

    P( t );
}
