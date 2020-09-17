#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using namespace sdot;

// // nsmake cxx_name nvcc
// // nsmake cpp_flag --expt-extended-lambda
// // nsmake cpp_flag --x
// // nsmake cpp_flag cu

int main() {
    using Arch = MachineArch::Native;
    // using Arch = MachineArch::Gpu;
    using TF = float;
    using TI = int;

    using Ep = SetOfElementaryPolytops<2,2,TF,TI,Arch>;
    using Pt = typename Ep::Pt;

    TI nb_shapes = 10;

    Ep ep;
    ep.add_shape( "3", { Pt{ 0, 0 }, Pt{ 1, 0 }, Pt{ 0, 1 } }, nb_shapes );

    // prepare the cuts
    Vec<TF,Arch> dirs[ 2 ];
    Vec<TF,Arch> sps;

    dirs[ 0 ].resize( nb_shapes );
    dirs[ 1 ].resize( nb_shapes );
    sps.resize( nb_shapes );

    for( TI i = 0; i < nb_shapes; ++i ) {
        TF a = 2 * M_PI * i / nb_shapes;
        dirs[ 0 ][ i ] = cos( a );
        dirs[ 1 ][ i ] = sin( a );
        sps[ i ] = ( cos( a ) + sin( a ) ) / 3;
    }

    // make the cuts
    ep.plane_cut( { dirs[ 0 ].ptr(), dirs[ 1 ].ptr() }, sps.ptr() );

    // VtkOutput vo;
    // ep.display_vtk( vo );
    // vo.save( "out.vtk" );
    // P( ep );
}

