#define DIM 2

#include "../src/sdot/support/VtkOutput.h"
#include "../src/sdot/Grids/ZGrid.h"
#include "../src/sdot/support/P.h"
#include <gtest/gtest.h>

#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )
using namespace sdot;

TEST( ZGrid, RegularCuts ) {
    std::vector<TF> xs, ys, ws;
    std::vector<ST> is;
    for( ST i = 0; i < 400; ++i ) {
        xs.push_back( 1.0 * rand() / RAND_MAX );
        ys.push_back( 1.0 * rand() / RAND_MAX );
        ws.push_back( 1.0 );
        is.push_back( i );
    }


    ZGrid zgrid;
    zgrid.update( [&]( const ZGrid::CbConstruct &cb ) {
        cb( { xs.data(), ys.data() }, ws.data(), is.data(), xs.size() );
    }, { .hist_min_point = { 0, 0 }, .hist_max_point = { 1, 1 } } );
 }
