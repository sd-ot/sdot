#include <parex/arch/SimdVec.h>
#include <parex/support/P.h>

using namespace parex;
using TF = double;
using TI = int;

int main() {
    using VF = SimdVec<TF>;
    VF u = VF( 17 ) + VF::iota();
    VF v = VF( 18 ) - VF::iota();
    P( u, v );
    P( min( u, v ) );
    P( max( u, v ) );
}
