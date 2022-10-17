#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Support/ThreadPool.h"
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class Grid,class Bounds,class Pt,class TF,class Func>
TF get_boundary_integral( Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func ) {
    using std::sqrt;
    using std::cos;
    using std::sin;

    // get connectivity of particles with ext surfaces. TODO: optimize
    struct DataPerThread {
        DataPerThread( std::size_t nb_diracs ) : neighbors( nb_diracs ), centroids( nb_diracs ), is_ext( nb_diracs, false ) {}
        std::vector<std::vector<std::size_t>> neighbors;
        std::vector<Pt>                       centroids;
        std::vector<bool>                     is_ext;
    };
    std::vector<DataPerThread> dpts( thread_pool.nb_threads(), nb_diracs );
    std::vector<int> thread_num_of_dirac( nb_diracs );

    // for each cell that has an ext boundary, get num_dirac of neighbors
    grid.for_each_laguerre_cell( [&]( auto &lc, auto num_dirac, int num_thread ) {
        thread_num_of_dirac[ num_dirac ] = num_thread;
        DataPerThread &dpt = dpts[ num_thread ];

        bool is_ext = false;
        bounds.for_each_intersection( lc, [&]( auto &cp, const auto &space_func ) {
            cp.for_each_boundary_item( space_func, radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
                if ( boundary_item.id == num_dirac )
                    is_ext = true;
            }, weights[ num_dirac ] );
        } );

        if ( is_ext ) {
            dpt.is_ext[ num_dirac ] = true;

            TF mass = 0;
            Pt centroid = TF( 0 );
            std::vector<std::size_t> &n_ng = dpt.neighbors[ num_dirac ];
            bounds.for_each_intersection( lc, [&]( auto &cp, const auto &space_func ) {
                cp.add_centroid_contrib( centroid, mass, space_func, radial_func.func_for_final_cp_integration(), weights[ num_dirac ] );

                cp.for_each_boundary_item( space_func, radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
                    if ( boundary_item.id != num_dirac && boundary_item.id != -1ul ) {
                        for( std::size_t i = 0; ; ++i ) {
                            if ( i == n_ng.size() ) {
                                n_ng.push_back( boundary_item.id );
                                break;
                            }
                            if ( n_ng[ i ] == boundary_item.id )
                                break;
                        }
                    }
                }, weights[ num_dirac ] );
            } );

            dpt.centroids[ num_dirac ] = centroid / TF( mass + ( mass == 0 ) );
        }
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );

    //
    TF res = 0;
    for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
        DataPerThread &dpt_m = dpts[ thread_num_of_dirac[ num_dirac ] ];

        if ( dpt_m.is_ext[ num_dirac ] ) {
            std::vector<std::size_t> &n_ng = dpt_m.neighbors[ num_dirac ];
            for( std::size_t n = 0; n < n_ng.size(); ++n ) {
                DataPerThread &dpt_n = dpts[ thread_num_of_dirac[ n_ng[ n ] ] ];
                if ( dpt_n.is_ext[ n_ng[ n ] ] == false ) {
                    n_ng[ n-- ] = n_ng.back();
                    n_ng.pop_back();
                }
            }

            if ( n_ng.size() == 2 ) {
                DataPerThread &dpt_0 = dpts[ thread_num_of_dirac[ n_ng[ 0 ] ] ];
                DataPerThread &dpt_1 = dpts[ thread_num_of_dirac[ n_ng[ 1 ] ] ];
                if ( dpt_0.is_ext[ n_ng[ 0 ] ] && dpt_1.is_ext[ n_ng[ 1 ] ] ) {
                    Pt p0 = dpt_0.centroids[ n_ng[ 0 ] ];
                    Pt pm = dpt_m.centroids[ num_dirac ];
                    Pt p1 = dpt_1.centroids[ n_ng[ 1 ] ];
                    res += ( norm_2( pm - p0 ) + norm_2( p1 - pm ) ) / 2;
                }
            }
        }
    }

    return res;
}

} // namespace sdot
