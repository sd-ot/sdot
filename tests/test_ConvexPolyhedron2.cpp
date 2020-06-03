#include "../src/sdot/ConvexPolyhedron/ConvexPolyhedron2.h"
#include "../src/sdot/support/VtkOutput.h"
#include "../src/sdot/support/P.h"
#include <gtest/gtest.h>
#include <cmath>

#define CP2 SDOT_CONCAT_TOKEN_2( ConvexPolyhedron2_, PROFILE )
using namespace sdot;

std::vector<ST> perm_iota( ST beg, ST end ) {
    ST size = end - beg;

    std::vector<ST> res;
    res.reserve( size );
    for( std::size_t i = 0; i < size; ++i )
        res.push_back( beg + i );

    for( std::size_t i = 0; i < size; ++i )
        std::swap( res[ i ], res[ i + rand() % ( size - i ) ] );

    return res;
}

TEST( CP2, RegularCuts ) {
    CP2 cp;

    // cut data
    for( ST m : perm_iota( 3, 300 ) ) {
        cp.init_as_box( { -10, -10 }, { +10, +10 }, 0 );
        EXPECT_EQ( cp.nb_vertices(), 4 );
        EXPECT_EQ( cp.nb_edges(), 4 );

        std::vector<TF> cx, cy, cs;
        std::vector<ST> ci;
        for( ST n : perm_iota( 0, m ) ) {
            double t = 2.0 * M_PI * n / m;
            cx.push_back( std::cos( t ) );
            cy.push_back( std::sin( t ) );
            ci.push_back( 10 + n );
            cs.push_back( 1 );
        }

        // make the cut
        const TF *p[ 2 ] = { cx.data(), cy.data() };
        cp.plane_cut( p, cs.data(), ci.data(), cx.size() );

        // checks
        cp.for_each_boundary( [&]( const CP2::Boundary &b ) {
            auto p = [&]( auto pt, int s ) {
                int a = std::round( std::atan2( pt[ 1 ], pt[ 0 ] ) * m / M_PI );
                return int( a + s + 2 * m ) % ( 2 * m );
            };
            int v = p( b.geometry.points[ 0 ], +1 );
            int w = p( b.geometry.points[ 1 ], -1 );
            EXPECT_EQ( b.cut_id, 10 + v / 2 );
            EXPECT_EQ( v, w );
        } );
    }

    // display
    VtkOutput vo;
    cp.display_vtk( vo );
    vo.save( "out.vtk" );
}
