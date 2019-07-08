#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Support/ThreadPool.h"
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class TF,class Pt,class Grid,class Bounds,class Func>
void get_distances_from_boundaries( TF *res, const Pt *points, std::size_t nb_points, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func, bool count_domain_boundaries ) {
    using std::min;

    std::vector<std::vector<TF>> distances( thread_pool.nb_threads() ); // distances for each thread
    for( auto &d : distances )
        d.resize( nb_points, std::numeric_limits<TF>::max() );

    grid.for_each_laguerre_cell( [&]( auto &lc, auto /*num_dirac*/, int num_thread ) {
        bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> /*space_func*/ ) {
            for( std::size_t n = 0; n < nb_points; ++n )
                distances[ num_thread ][ n ] = min( distances[ num_thread ][ n ], cp.distance( points[ n ], count_domain_boundaries ) );
        } );
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );

    for( std::size_t n = 0; n < nb_points; ++n ) {
        res[ n ] = std::numeric_limits<TF>::max();
        for( std::size_t i = 0; i < distances.size(); ++i )
            res[ n ] = min( res[ n ], distances[ i ][ n ] );
    }
}

} // namespace sdot
