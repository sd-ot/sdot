#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;
using namespace sdot;

//// nsmake cxx_name nvcc
//// nsmake cpp_flag --expt-extended-lambda
//// nsmake cpp_flag --x
//// nsmake cpp_flag cu

int main() {
    using Ep = SetOfElementaryPolytops<2,2,float,int,MachineArch::Gpu>;
    using Pt = typename Ep::Pt;

    Ep ep;
    ep.add_shape( "3", 2, { Pt{ 0, 0 }, Pt{ 1, 0 }, Pt{ 0, 1 } }, 0, 10 );



    VtkOutput vo;
    ep.display_vtk( vo );
    vo.save( "out.vtk" );
    P( ep );
}

