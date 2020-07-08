#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

//// nsmake cxx_name nvcc
//// nsmake cpp_flag --x
//// nsmake cpp_flag cu

int main() {
    Vec<float,CpuArch::Gpu> v( 10 );
    P( v );
}

