#include "../src/sdot/ConvexPolyhedron/ConvexPolyhedron2.h"
#include "../src/sdot/support/Time.h"
#include "../src/sdot/support/P.h"

#define CP2 SDOT_CONCAT_TOKEN_2( ConvexPolyhedron2_, PROFILE )
using namespace sdot;

void __attribute__ ((noinline)) init( CP2 &cp, const TF **/*p*/, const TF */*cs*/, const ST */*ci*/, ST /*nb_cuts*/ ) {
    cp.init_as_box( { -10, -10 }, { +10, +10 }, 0 );
}

void __attribute__ ((noinline)) pcut( CP2 &cp, const TF **p, const TF *cs, const ST *ci, ST nb_cuts ) {
    cp.init_as_box( { -10, -10 }, { +10, +10 }, 0 );
    cp.plane_cut( p, cs, ci, nb_cuts );
}

int main( int /*argc*/, char **/*argv*/ ) {
    // cut list
    std::vector<TF> cx, cy, cs;
    std::vector<ST> ci;
    for( ST n = 0, m = 5; n < m; ++n ) {
        double t = 2.0 * M_PI * n / m;
        cx.push_back( std::cos( t ) );
        cy.push_back( std::sin( t ) );
        ci.push_back( 10 + n );
        cs.push_back( 1 );
    }

    const TF *p[ 2 ] = {
        cx.data(),
        cy.data()
    };

    std::uint64_t nb_cycles_init = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t nb_cycles_pcut = std::numeric_limits<std::uint64_t>::max();

    // convex polyhedron
    CP2 cp;
    for( std::uint64_t r = 0; r < 1000; ++r ) {
        for( std::uint64_t rep = 0; rep < 100; ++rep ) {
            std::uint64_t b, e;
            RDTSC_START( b );
            init( cp, p, cs.data(), ci.data(), cx.size() );
            RDTSC_FINAL( e );
            nb_cycles_init = std::min( nb_cycles_init, e - b );
        }
        for( std::uint64_t rep = 0; rep < 100; ++rep ) {
            std::uint64_t b, e;
            RDTSC_START( b );
            pcut( cp, p, cs.data(), ci.data(), cx.size() );
            RDTSC_FINAL( e );
            nb_cycles_pcut = std::min( nb_cycles_pcut, e - b );
        }
    }

    // P( cp );
    P( double( nb_cycles_pcut - nb_cycles_init ) / cx.size() );
}
