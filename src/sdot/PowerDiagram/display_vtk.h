#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Display/VtkOutput.h"
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class TF,class Grid,class Bounds,class Pt,class Func>
void display_vtk( VtkOutput<1,TF> &res, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func ) {
    grid.for_each_laguerre_cell( [&]( auto &lc, auto num_dirac, int ) {
        bounds.for_each_intersection( lc, [&]( auto &cp, const auto &/*space_func*/ ) {
            cp.display( res, { TF( num_dirac ) } );
        } );
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );
}

template<class TF,class Grid,class Bounds,class Pt>
void display_vtk( VtkOutput<1,TF> &res, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs ) {
    display_vtk( res, grid, bounds, positions, weights, nb_diracs, FunctionEnum::Unit() );
}

} // namespace sdot
