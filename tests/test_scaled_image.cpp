#include "../src/sdot/PowerDiagram/Visitors/SpZGrid.h"
#include "../src/sdot/PowerDiagram/get_integrals.h"
#include "../src/sdot/PowerDiagram/display_vtk.h"
#include "../src/sdot/Domains/ScaledImage.h"
#include "../src/sdot/Support/Stream.h"
// #include "catch_main.h"

//// nsmake cpp_flag -march=native
using namespace sdot;

int main() {
    struct Pc { enum { dim = 2, allow_ball_cut = 0, allow_translations = 0 }; using TI = std::size_t; using TF = double; };
    using Bounds = ScaledImage<Pc>;
    using Grid = SpZGrid<Pc>;
    using std::pow;
    using std::cos;
    using std::sin;

    using Pt = typename Grid::Pt;
    using CP = typename Grid::CP;
    using TF = typename Pc::TF;
    using TI = typename Pc::TI;

    Grid grid;

    TI n = 20;
    std::vector<TF> img_data( n * n, TF( 1 ) );
    Bounds bounds( { 0, 0 }, { 1, 1 }, img_data.data(), { n, n } );

    std::vector<Pt> positions{ Pt{ 0.25, 0.25 }, Pt{ 0.75, 0.25 }, Pt{ 0.25, 0.75 }, Pt{ 0.75, 0.75 } };
    std::vector<TF> weights{ 1.0, 1.0, 1.0, 1.0 };

    std::vector<TF> integrals( weights.size() );
    grid.update( positions.data(), weights.data(), weights.size() );
    get_integrals( integrals.data(), grid, bounds, positions.data(), weights.data(), weights.size() );
    P( integrals ); // -> 0.25 0.25 0.25 0.25

    VtkOutput<1,TF> vo( { "num" } );
    display_vtk( vo, grid, bounds, positions.data(), weights.data(), weights.size() );
    vo.save( "lc.vtk" );
}
