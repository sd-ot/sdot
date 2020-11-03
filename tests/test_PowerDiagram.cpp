#include "../src/sdot/geometry/PowerDiagram.h"
#include "../src/sdot/kernels/KernelSlot.h"
#include "../src/sdot/support/P.h"

using namespace sdot;

using TI = std::uint64_t;
using TF = double;

void test_fill() {
    std::vector<std::unique_ptr<KernelSlot>> ak = KernelSlot::available_slots( TypeName<TF>::name(), TypeName<TI>::name() );
    // KernelSlot *ks = ak[ 0 ].get();

    PowerDiagram pd;

    pd.fill( [&]( auto f ) {
        f();
    } );
}

int main() {
    test_fill();
}
