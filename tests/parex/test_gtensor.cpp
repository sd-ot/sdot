#include <asimd/operations/assign_scalar.h>
#include <asimd/allocators/GpuAllocator.h>
#include <parex/containers/gtensor.h>
#include <parex/P.h>

// using namespace parex;

/// nsmake cxx_name nvcc

int main() {
    using Al = asimd::AlignedAllocator<double,64>;
    Al al;

    gtensor<double,3,Al> t;
    t.resize( al, 2, 3, 4 );
    P( t.shape() );

    for( std::size_t i = 0; i < 8*3*2; ++i )
        t.data()[ i ] = i;

    asimd::assign_scalar( t.data( 0, 0, 0 ), 17.0, 4 );
    P( t );
}
