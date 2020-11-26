#include <asimd/internal/P.h>
#include <asimd/SimdVec.h>

using namespace asimd;
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
