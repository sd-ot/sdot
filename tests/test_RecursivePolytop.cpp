#include "../src/sdot/geometry/RecursivePolytop.h"
#include "../src/sdot/support/Rational.h"
#include "../src/sdot/support/P.h"
using TI = std::size_t;

//// nsmake cpp_flag -march=native

int main() {
    using TF = Rational;
    using Rp = RecursivePolytop<TF,2>;

    Rp rp = Rp::convex_hull( {
        Rp::Node{ { 0, 0 }, 0 },
        Rp::Node{ { 1, 0 }, 1 },
        Rp::Node{ { 0, 1 }, 2 },
    } );

    P( rp );
}
