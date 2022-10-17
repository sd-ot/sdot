#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Support/ThreadPool.h"
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class Pt,class Grid,class Bounds,class TF,class Func>
void get_der_boundary_integral( Pt *res, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func ) {
    using std::sqrt;
    using std::cos;
    using std::sin;

    // get connectivity of particles with ext surfaces. TODO: optimize
    for( std::size_t n = 0; n < nb_diracs; ++n )
        for( std::size_t d = 0; d < Grid::dim; ++d )
            res[ n ][ d ] = TF( 0 );

    // for each cell that has an ext boundary, get num_dirac of neighbors
    grid.for_each_laguerre_cell( [&]( auto &lc, auto num_dirac, int num_thread ) {
        bounds.for_each_intersection( lc, [&]( auto &cp, const auto &space_func ) {
            cp.for_each_boundary_item( space_func, radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
                if ( boundary_item.id == num_dirac ) {
                    TF R = sqrt( weights[ num_dirac ] );
                    res[ num_dirac ].x += R * ( sin( boundary_item.a0 ) - sin( boundary_item.a1 ) );
                    res[ num_dirac ].y += R * ( cos( boundary_item.a1 ) - cos( boundary_item.a0 ) );
                }
            }, weights[ num_dirac ] );
        } );
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );
}

//template<class Grid,class Bounds,class Pt,class TF,class Func>
//void get_der_boundary_integral( TF *res, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func ) {
//    using std::sqrt;
//    using std::cos;
//    using std::sin;

//    // get connectivity of particles with ext surfaces. TODO: optimize
//    struct DataPerThread {
//        DataPerThread( std::size_t nb_diracs ) : neighbors( nb_diracs ), centroids( nb_diracs ), is_ext( nb_diracs, false ) {}
//        std::vector<std::vector<std::size_t>> neighbors;
//        std::vector<Pt>                       centroids;
//        std::vector<bool>                     is_ext;
//    };
//    std::vector<DataPerThread> dpts( thread_pool.nb_threads(), nb_diracs );
//    std::vector<int> thread_num_of_dirac( nb_diracs );

//    // for each cell that has an ext boundary, get num_dirac of neighbors
//    grid.for_each_laguerre_cell( [&]( auto &lc, auto num_dirac, int num_thread ) {
//        thread_num_of_dirac[ num_dirac ] = num_thread;
//        DataPerThread &dpt = dpts[ num_thread ];

//        bool is_ext = false;
//        bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> space_func ) {
//            cp.for_each_boundary_item( radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
//                if ( boundary_item.id == num_dirac )
//                    is_ext = true;
//            }, weights[ num_dirac ] );
//        } );

//        if ( is_ext ) {
//            dpt.is_ext[ num_dirac ] = true;

//            TF mass = 0;
//            Pt centroid = TF( 0 );
//            std::vector<std::size_t> &n_ng = dpt.neighbors[ num_dirac ];
//            bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> space_func ) {
//                cp.add_centroid_contrib( centroid, mass, radial_func.func_for_final_cp_integration(), space_func, weights[ num_dirac ] );

//                cp.for_each_boundary_item( radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
//                    if ( boundary_item.id != num_dirac && boundary_item.id != -1ul ) {
//                        for( std::size_t i = 0; ; ++i ) {
//                            if ( i == n_ng.size() ) {
//                                n_ng.push_back( boundary_item.id );
//                                break;
//                            }
//                            if ( n_ng[ i ] == boundary_item.id )
//                                break;
//                        }
//                    }
//                }, weights[ num_dirac ] );
//            } );

//            dpt.centroids[ num_dirac ] = centroid / TF( mass + ( mass == 0 ) );
//        }
//    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );

//    //
//    for( std::size_t n = 0; n < Grid::dim * nb_diracs; ++n )
//        res[ n ] = 0;

//    //
//    for( std::size_t num_dirac_0 = 0; num_dirac_0 < nb_diracs; ++num_dirac_0 ) {
//        DataPerThread &dpt_0 = dpts[ thread_num_of_dirac[ num_dirac_0 ] ];
//        if ( dpt_0.is_ext[ num_dirac_0 ] ) {
//            std::vector<std::size_t> &n_ng = dpt_0.neighbors[ num_dirac_0 ];
//            for( std::size_t nn = 0; nn < n_ng.size(); ++nn ) {
//                std::size_t num_dirac_1 = n_ng[ nn ];
//                DataPerThread &dpt_1 = dpts[ thread_num_of_dirac[ num_dirac_1 ] ];
//                if ( dpt_1.is_ext[ n_ng[ nn ] ] == false ) {
//                    n_ng[ nn-- ] = n_ng.back();
//                    n_ng.pop_back();
//                }
//            }

//            if ( n_ng.size() == 2 ) {
//                std::size_t num_dirac_a = n_ng[ 0 ];
//                std::size_t num_dirac_b = n_ng[ 1 ];
//                DataPerThread &dpt_a = dpts[ thread_num_of_dirac[ num_dirac_a ] ];
//                DataPerThread &dpt_b = dpts[ thread_num_of_dirac[ num_dirac_b ] ];
//                Pt p0 = dpt_0.centroids[ num_dirac_0 ];
//                Pt pa = dpt_a.centroids[ num_dirac_a ];
//                Pt pb = dpt_b.centroids[ num_dirac_b ];

//                TF da = norm_2( pa - p0 );
//                TF db = norm_2( pb - p0 );
//                Pt lo = ( pa - p0 ) / da + ( pb - p0 ) / db;
//                for( std::size_t d = 0; d < Grid::dim; ++d )
//                    res[ Grid::dim * num_dirac_0 + d ] += lo[ d ];
//            }
//        }
//    }
//}

} // namespace sdot
