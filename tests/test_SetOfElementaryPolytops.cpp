#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/kernels/KernelSlot.h"
#include "../src/sdot/support/P.h"
using namespace sdot;

int main() {
    std::vector<std::unique_ptr<KernelSlot>> ak = KernelSlot::available_slots();
    KernelSlot *ks = ak[ 0 ].get();

    SetOfElementaryPolytops sp( ks, 2 );
    sp.add_repeated( triangle(), 10, { ks, std::vector<double>{ 0, 0,  1, 0,  0, 1 } } );
    sp.add_repeated( triangle(), 10, { ks, std::vector<double>{ 1, 2,  3, 4,  5, 6 } } );
    P( sp );

    //sp.display_vtk( vo );
}
