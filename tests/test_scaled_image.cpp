#include "../src/sdot/Support/Stream.h"

#include "../src/sdot/PowerDiagram/Visitors/SpZGrid.h"
#include "../src/sdot/PowerDiagram/get_integrals.h"
#include "../src/sdot/PowerDiagram/display_vtk.h"
#include "../src/sdot/Domains/ScaledImage.h"
#include "../src/sdot/Support/Stream.h"
#include "catch_main.h"

using namespace sdot;

//int main() {
//    struct Pc { enum { dim = 2, allow_ball_cut = 0, allow_translations = 0 }; using TI = std::size_t; using TF = double; };
//    using Bounds = ScaledImage<Pc>;
//    using Grid = SpZGrid<Pc>;
//    using std::pow;
//    using std::cos;
//    using std::sin;

//    using Pt = typename Grid::Pt;
//    using CP = typename Grid::CP;
//    using TF = typename Pc::TF;
//    using TI = typename Pc::TI;

//    Grid grid;

//    TI n = 20;
//    std::vector<TF> img_data( n * n, TF( 1 ) );
//    Bounds bounds( { 0, 0 }, { 1, 1 }, img_data.data(), { n, n } );

//    std::vector<Pt> positions{ Pt{ 0.25, 0.25 }, Pt{ 0.75, 0.25 }, Pt{ 0.25, 0.75 }, Pt{ 0.75, 0.75 } };
//    std::vector<TF> weights{ 1.0, 1.0, 1.0, 1.0 };

//    std::vector<TF> integrals( weights.size() );
//    grid.update( positions.data(), weights.data(), weights.size() );
//    get_integrals( integrals.data(), grid, bounds, positions.data(), weights.data(), weights.size() );
//    P( integrals ); // -> 0.25 0.25 0.25 0.25

//    VtkOutput<1,TF> vo( { "num" } );
//    display_vtk( vo, grid, bounds, positions.data(), weights.data(), weights.size() );
//    vo.save( "lc.vtk" );
//}

struct Pc { enum { dim = 2, allow_ball_cut = 0, allow_translations = 0 }; using TI = std::size_t; using TF = double; using CI = std::size_t; }; // boost::multiprecision::mpfr_float_100
using Bounds = ScaledImage<Pc>;
using Grid = SpZGrid<Pc>;
using std::pow;
using std::cos;
using std::sin;

using Pt = typename Grid::Pt;
using CP = typename Grid::CP;
using TF = typename Pc::TF;
using TI = typename Pc::TI;

TEST_CASE( "scaled image 1", "" ) {
    Grid grid;
    std::vector<TF> img_data{ 2, 3, 4, 5, 6, 7 };
    Bounds bounds( { 0, 0 }, { 1, 1 }, img_data.data(), { 1, 1 }, 6 );

    std::vector<Pt> positions{ Pt{ 0.5, 0.5 } };
    std::vector<TF> weights{ 1.0 };

    std::vector<TF> integrals( weights.size() );
    grid.update( positions.data(), weights.data(), weights.size() );
    get_integrals( integrals.data(), grid, bounds, positions.data(), weights.data(), weights.size() );

    // Integrate[ Integrate[ 2 + 3 * x + 4 * y + 5 * x * x + 6 * x * y + 7 * y * y, { x, 0, 1 } ], { y, 0, 1 } ]
    CHECK( abs( integrals[ 0 ] - 11 ) < 1e-6 );
}

TEST_CASE( "scaled image 2", "" ) {
    Grid grid;
    std::vector<TF> img_data{ 2, 3, 4, 5, 6, 7 };
    Bounds bounds( { 0, 0 }, { 1, 1 }, img_data.data(), { 1, 1 }, 6 );

    std::vector<Pt> positions{ Pt{ 0.25, 0.25 }, Pt{ 0.75, 0.25 }, Pt{ 0.25, 0.75 }, Pt{ 0.75, 0.75 } };
    std::vector<TF> weights{ 1.0, 1.0, 1.0, 1.0 };

    std::vector<TF> integrals( weights.size() );
    grid.update( positions.data(), weights.data(), weights.size() );
    get_integrals( integrals.data(), grid, bounds, positions.data(), weights.data(), weights.size() );

    // Integrate[ Integrate[ 2 + 3 * x + 4 * y + 5 * x * x + 6 * x * y + 7 * y * y, { x, 0.0, 0.5 } ], { y, 0.0, 0.5 } ] -> 1.28125
    // Integrate[ Integrate[ 2 + 3 * x + 4 * y + 5 * x * x + 6 * x * y + 7 * y * y, { x, 0.5, 1.0 } ], { y, 0.0, 0.5 } ] -> 2.46875
    // Integrate[ Integrate[ 2 + 3 * x + 4 * y + 5 * x * x + 6 * x * y + 7 * y * y, { x, 0.0, 0.5 } ], { y, 0.5, 1.0 } ] -> 2.84375
    // Integrate[ Integrate[ 2 + 3 * x + 4 * y + 5 * x * x + 6 * x * y + 7 * y * y, { x, 0.5, 1.0 } ], { y, 0.5, 1.0 } ] -> 4.40625
    CHECK( abs( integrals[ 0 ] - 1.28125 ) < 1e-6 );
    CHECK( abs( integrals[ 1 ] - 2.46875 ) < 1e-6 );
    CHECK( abs( integrals[ 2 ] - 2.84375 ) < 1e-6 );
    CHECK( abs( integrals[ 3 ] - 4.40625 ) < 1e-6 );
}
