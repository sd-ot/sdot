#include "../src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../src/sdot/PowerDiagram/Visitors/SpZGrid.h"
#include "catch_main.h"

//// nsmake cpp_flag -march=native
using namespace sdot;
using std::abs;

TEST_CASE( "ZGrid measures" ) {
    struct Pc {
        enum { dim = 2, allow_ball_cut = false, allow_translations = true };
        using TI = std::size_t;
        using TF = double;
        using CI = int;
    };
    using Bounds = ConvexPolyhedronAssembly<Pc>;
    using Grid   = SpZGrid<Pc>;

    std::vector<Grid::Pt> positions;
    std::vector<Grid::TF> weights;
    for( std::size_t i = 0; i < 10; ++i ) {
        positions.push_back( { 1.0 * rand() / RAND_MAX, 10.0 * rand() / RAND_MAX } );
        weights.push_back( 1.0 );
    }

    Bounds bounds;
    bounds.add_box( { -1, 0 }, { 2, 10 } );

    Grid grid( 11 );
    grid.translations.push_back( { +1, 0 } );
    grid.translations.push_back( { -1, 0 } );
    grid.update( positions.data(), weights.data(), positions.size() );

    VtkOutput<1> vo( { "num" } );
    std::mutex mutex;
    grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac_0 ) {
        bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
            mutex.lock();
            cp.display( vo, { 1.0 * num_dirac_0 } );
            mutex.unlock();
        } );
    }, bounds.englobing_convex_polyhedron(), positions.data(), weights.data(), positions.size() );

    grid.display( vo, 0.1 );
    vo.save( "vtk/pd.vtk" );
}

